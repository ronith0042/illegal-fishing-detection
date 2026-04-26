from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.cluster import DBSCAN
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    (ROOT / "data" / "trollers.csv", "Troller"),
    (ROOT / "data" / "pole_and_line.csv", "Pole and line"),
]
OUTPUT = ROOT / "NEREUS_ANOMALY_MAP.html"
MODELS_DIR = ROOT / "models"

MAX_PINS_PER_LAYER = 900
MAX_TRACK_POINTS_PER_VESSEL = 450

RISK_ZONES = [
    {
        "name": "Argentine Blue Hole / Mile 201 risk area",
        "lat": -45.0,
        "lon": -60.0,
        "radius_km": 520,
        "color": "#d72638",
    },
    {
        "name": "Galapagos Marine Reserve buffer",
        "lat": -0.8,
        "lon": -90.6,
        "radius_km": 360,
        "color": "#2d9cdb",
    },
    {
        "name": "Canary-Madeira high-loitering corridor",
        "lat": 31.8,
        "lon": -16.7,
        "radius_km": 350,
        "color": "#f2994a",
    },
]

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))

def load_data() -> pd.DataFrame:
    frames = []
    for csv_path, gear_type in DATASETS:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame["gear_type"] = gear_type
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No CSV files found in data/")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates()
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    df = df[df["speed"].between(0, 60)]
    df["mmsi"] = df["mmsi"].astype("int64").astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp", "lat", "lon"])
    return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

def calculate_rot(df):
    df = df.sort_values(["mmsi", "timestamp"])
    time_delta = df.groupby("mmsi")["timestamp"].diff().dt.total_seconds()
    course_diff = df.groupby("mmsi")["course"].diff().abs()
    course_diff = np.minimum(course_diff, 360 - course_diff)
    rot = course_diff / (time_delta / 60.0 + 1e-6)
    return rot.fillna(0), time_delta.fillna(0)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    previous = df.groupby("mmsi")[["lat", "lon", "timestamp", "course"]].shift(1)
    df["prev_lat"] = previous["lat"]
    df["prev_lon"] = previous["lon"]
    df["time_gap_hours"] = (df["timestamp"] - previous["timestamp"]).dt.total_seconds() / 3600
    df["step_km"] = haversine_km(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df["derived_speed_knots"] = df["step_km"] / (df["time_gap_hours"] + 1e-6) / 1.852
    df.loc[~np.isfinite(df["derived_speed_knots"]), "derived_speed_knots"] = np.nan
    df["course_change"] = (df["course"] - previous["course"]).abs()
    df["course_change"] = np.minimum(df["course_change"], 360 - df["course_change"])

    distances = []
    nearest_names = []
    for _, zone in pd.DataFrame(RISK_ZONES).iterrows():
        distances.append(haversine_km(df["lat"], df["lon"], zone["lat"], zone["lon"]))
    distance_matrix = np.vstack(distances).T
    nearest_index = np.nanargmin(distance_matrix, axis=1)
    zone_names = [zone["name"] for zone in RISK_ZONES]
    for idx in nearest_index:
        nearest_names.append(zone_names[idx])
    
    df["nearest_zone"] = nearest_names
    df["nearest_zone_km"] = distance_matrix[np.arange(len(df)), nearest_index]
    df["inside_risk_zone"] = [
        distance_matrix[i, nearest_index[i]] <= RISK_ZONES[nearest_index[i]]["radius_km"]
        for i in range(len(df))
    ]
    df["near_risk_zone"] = [
        distance_matrix[i, nearest_index[i]] <= RISK_ZONES[nearest_index[i]]["radius_km"] + 80
        for i in range(len(df))
    ]

    df["dark_gap"] = df["time_gap_hours"] >= 6
    df["tactical_darkness"] = df["dark_gap"] & (df["near_risk_zone"] | (df["distance_from_shore"] <= 25000))
    df["gps_jump"] = (df["derived_speed_knots"] >= 45) & (df["time_gap_hours"] >= 0.05)
    df["spoofing_risk"] = df["gps_jump"] | ((df["course_change"] >= 150) & (df["speed"] >= 15))
    df["loitering"] = (
        (df["speed"] <= 2)
        & (df["time_gap_hours"].between(0.02, 4))
        & (df["step_km"] <= 1.0)
        & (df["distance_from_port"] > 10000)
    )
    df["fishing_zone_risk"] = (df["is_fishing"] > 0) & (df["inside_risk_zone"] | (df["distance_from_shore"] <= 15000))

    score = np.zeros(len(df), dtype=int)
    score += np.where(df["tactical_darkness"], 50, 0)
    score += np.where(df["gps_jump"], 35, 0)
    score += np.where(df["loitering"], 15, 0)
    score += np.where(df["fishing_zone_risk"], 35, 0)
    score += np.where(df["inside_risk_zone"], 100, 0)
    df["risk_score"] = score
    return df

def detect_hidden_intrusions(df, model, scaler):
    print("Running Bi-LSTM path reconstruction (The Truth Engine)...")
    FEATURE_COLS = ["lat", "lon", "speed", "course", "rot", "time_delta"]
    df["rot"], df["time_delta"] = calculate_rot(df)
    df["hidden_intrusion"] = False
    gaps = df[df["dark_gap"]].copy()
    if gaps.empty: return []
    reconstructed_intrusions = []
    for idx, gap_row in tqdm(gaps.iterrows(), total=len(gaps), desc="Reconstructing paths"):
        mmsi = gap_row["mmsi"]
        vessel_data = df[df["mmsi"] == mmsi].sort_values("timestamp")
        try:
            gap_idx = vessel_data.index.get_loc(idx)
        except (KeyError, ValueError): continue
        if gap_idx < 5 or gap_idx + 5 >= len(vessel_data): continue
        context = vessel_data.iloc[gap_idx - 5 : gap_idx + 6].copy()
        scaled_input = scaler.transform(context[FEATURE_COLS])
        scaled_input[5, :] = 0 # Mask middle
        input_seq = scaled_input.reshape(1, 11, 6)
        pred_scaled = model.predict(input_seq, verbose=0)
        dummy = np.zeros((1, 6))
        dummy[0, :2] = pred_scaled[0]
        p_lat, p_lon = scaler.inverse_transform(dummy)[0, :2]
        for zone in RISK_ZONES:
            if haversine_km(p_lat, p_lon, zone["lat"], zone["lon"]) <= zone["radius_km"]:
                reconstructed_intrusions.append({"mmsi": mmsi, "timestamp": context.iloc[5]["timestamp"], "lat": p_lat, "lon": p_lon, "zone": zone["name"]})
                df.at[idx, "hidden_intrusion"] = True
                df.at[idx, "risk_score"] += 100
                break
    return reconstructed_intrusions

def detect_rendezvous(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slow = df[(df["speed"] <= 2) & (df["distance_from_port"] > 50000) & (df["distance_from_shore"] > 10000)].copy()
    if slow.empty: return pd.DataFrame(), pd.DataFrame()
    coords = np.radians(slow[["lat", "lon"]].values)
    db = DBSCAN(eps=0.5/6371.0, min_samples=2, metric="haversine").fit(coords)
    slow["cluster"] = db.labels_
    clusters_df = slow[slow["cluster"] != -1].copy()
    if clusters_df.empty: return pd.DataFrame(), pd.DataFrame()
    rendezvous = clusters_df.groupby("cluster").agg(
        lat=("lat", "mean"), lon=("lon", "mean"),
        start=("timestamp", "min"), end=("timestamp", "max"),
        points=("mmsi", "size"), vessels=("mmsi", "nunique"),
        vessel_ids=("mmsi", lambda x: ", ".join(sorted(set(x.astype(str)))[:6])),
    ).reset_index()
    rendezvous["duration_hours"] = (rendezvous["end"] - rendezvous["start"]).dt.total_seconds() / 3600
    rendezvous = rendezvous[(rendezvous["vessels"] >= 2) & (rendezvous["duration_hours"] >= 2)]
    final_clusters = clusters_df[clusters_df["cluster"].isin(rendezvous["cluster"])]
    rendezvous["day"] = rendezvous["start"].dt.strftime("%Y-%m-%d")
    return rendezvous.sort_values(["vessels", "duration_hours", "points"], ascending=False), final_clusters

def sample_rows(df: pd.DataFrame, mask: pd.Series, limit: int) -> pd.DataFrame:
    rows = df[mask]
    if len(rows) > limit:
        # Use a random sample to ensure geographic diversity across the map
        return rows.sample(n=limit, random_state=42)
    return rows

def popup_for_row(row: pd.Series) -> str:
    reasons = []
    if row.get("hidden_intrusion", False): reasons.append(f"<b style='color:#dc2626'>HIDDEN ZONE INTRUSION</b> (Bi-LSTM)")
    if row["tactical_darkness"]: reasons.append(f"Tactical AIS darkness: {row['time_gap_hours']:.1f} h signal gap")
    elif row["dark_gap"]: reasons.append(f"AIS dark gap: {row['time_gap_hours']:.1f} h")
    if row["gps_jump"]: reasons.append(f"GPS jump/spoofing risk: {row['derived_speed_knots']:.1f} kn")
    if row["loitering"]: reasons.append(f"Low-speed loitering: {row['speed']:.1f} kn")
    if row["fishing_zone_risk"]: reasons.append("Fishing in risk area")
    if row["inside_risk_zone"]: reasons.append(f"Inside {row['nearest_zone']}")
    if row.get("rendezvous_event", False): reasons.append("Rendezvous candidate")
    reason_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in reasons)
    return f"<b>MMSI:</b> {row['mmsi']}<br><b>Risk:</b> {int(row['risk_score'])}<br><b>Anomalies:</b><ul>{reason_html}</ul>"

def build_map_payload(df: pd.DataFrame, rendezvous: pd.DataFrame) -> dict:
    layers = {
        "Hidden Zone Intrusions (AI)": marker_payload(df[df.get("hidden_intrusion", False)], "#dc2626"),
        "Tactical AIS darkness": marker_payload(sample_rows(df, df["tactical_darkness"], MAX_PINS_PER_LAYER), "#7b2cbf"),
        "GPS jump / spoofing": marker_payload(sample_rows(df, df["spoofing_risk"], MAX_PINS_PER_LAYER), "#111827"),
        "Low-speed loitering": marker_payload(sample_rows(df, df["loitering"], MAX_PINS_PER_LAYER), "#f59e0b"),
        "Fishing in risk area": marker_payload(sample_rows(df, df["fishing_zone_risk"], MAX_PINS_PER_LAYER), "#ef4444"),
    }
    
    rendezvous_markers = []
    for _, row in rendezvous.head(MAX_PINS_PER_LAYER).iterrows():
        popup = f"<b>Rendezvous</b><br><b>Duration:</b> {row['duration_hours']:.1f}h<br><b>Vessels:</b> {int(row['vessels'])}"
        rendezvous_markers.append({"lat": round(float(row["lat"]), 6), "lon": round(float(row["lon"]), 6), "color": "#10b981", "score": 40, "popup": popup})
    layers["Rendezvous clusters"] = rendezvous_markers

    tracks = []
    top_mmsis = df.groupby("mmsi")["risk_score"].sum().sort_values(ascending=False).head(8).index.tolist()
    for mmsi in top_mmsis:
        vessel = df[df["mmsi"] == mmsi].sort_values("timestamp")
        if len(vessel) > MAX_TRACK_POINTS_PER_VESSEL:
            vessel = vessel.iloc[np.linspace(0, len(vessel) - 1, MAX_TRACK_POINTS_PER_VESSEL).astype(int)]
        tracks.append({"mmsi": str(mmsi), "points": [[round(float(r.lat), 6), round(float(r.lon), 6)] for r in vessel.itertuples()]})

    # Radar Data for top 5
    radar_data = []
    for mmsi in top_mmsis[:5]:
        v = df[df["mmsi"] == mmsi]
        radar_data.append({
            "mmsi": str(mmsi),
            "dark": int(v["dark_gap"].sum()),
            "intrusion": int(v["inside_risk_zone"].sum() + v.get("hidden_intrusion", 0).sum()),
            "activity": int((v["is_fishing"] > 0).sum() + (v["loitering"]).sum()),
            "rendezvous": int(v.get("rendezvous_event", 0).sum())
        })

    # Timeline Data for top vessel
    top_vessel = top_mmsis[0] if top_mmsis else None
    timeline = []
    if top_vessel:
        v_full = df[df["mmsi"] == top_vessel].sort_values("timestamp")
        # To avoid massive payload, we take 200 samples
        v_sample = v_full.iloc[np.linspace(0, len(v_full)-1, 200).astype(int)] if len(v_full) > 200 else v_full
        for _, r in v_sample.iterrows():
            timeline.append({
                "t": r["timestamp"].isoformat(),
                "v": 0 if r["dark_gap"] else 1,
                "a": 1 if r["risk_score"] > 50 else 0
            })

    summary = {
        "rows": int(len(df)), "vessels": int(df["mmsi"].nunique()),
        "dark": int(df["dark_gap"].sum()), "tactical_dark": int(df["tactical_darkness"].sum()),
        "gps": int(df["spoofing_risk"].sum()), "loiter": int(df["loitering"].sum()),
        "fishing": int(df["fishing_zone_risk"].sum()), "rendezvous": int(len(rendezvous)),
        "hidden_intrusions": int(df.get("hidden_intrusion", pd.Series([False]*len(df))).sum()),
        "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max()),
    }
    return {
        "center": [float(df["lat"].median()), float(df["lon"].median())], 
        "layers": layers, "tracks": tracks, "zones": RISK_ZONES, "summary": summary,
        "radar": radar_data, "timeline": timeline, "top_mmsi": str(top_vessel)
    }

def marker_payload(rows: pd.DataFrame, color: str) -> list[dict]:
    payload = []
    for _, row in rows.iterrows():
        payload.append({"lat": round(float(row["lat"]), 6), "lon": round(float(row["lon"]), 6), "color": color, "score": int(row["risk_score"]), "popup": popup_for_row(row)})
    return payload

def write_html(payload: dict) -> None:
    payload_json = json.dumps(payload)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Nereus Forensic Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; }}
    #map {{ height: 100vh; width: 100%; }}
    .overlay-panel {{ position: absolute; z-index: 1000; top: 20px; left: 20px; width: 380px; background: rgba(15, 23, 42, 0.9); padding: 20px; border-radius: 12px; border: 1px solid #334155; backdrop-filter: blur(8px); max-height: 90vh; overflow-y: auto; }}
    .chart-container {{ margin-top: 20px; background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 8px; }}
    h1 {{ font-size: 20px; margin: 0 0 10px; color: #38bdf8; }}
    .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }}
    .stat-card {{ background: #1e293b; padding: 10px; border-radius: 6px; text-align: center; border-left: 3px solid #38bdf8; }}
    .stat-val {{ font-size: 18px; font-weight: bold; display: block; }}
    .stat-lbl {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
    .timeline-bar {{ height: 40px; width: 100%; display: flex; background: #334155; border-radius: 4px; overflow: hidden; margin-top: 10px; }}
    .bar-seg {{ height: 100%; }}
    .leaflet-popup-content-wrapper {{ background: #1e293b; color: #f8fafc; }}
    .leaflet-popup-tip {{ background: #1e293b; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="overlay-panel">
    <h1>Project Nereus: Forensic AI</h1>
    <div class="stat-grid" id="stats"></div>
    
    <div class="chart-container">
      <h2 style="font-size:14px; margin-top:0;">Forensic Radar: Top Suspects</h2>
      <canvas id="radarChart"></canvas>
    </div>

    <div class="chart-container">
      <h2 style="font-size:14px; margin-top:0;">Transmission Timeline (MMSI: <span id="topMmsi"></span>)</h2>
      <canvas id="timelineChart" style="height: 120px;"></canvas>
      <p style="font-size:10px; color:#94a3b8; margin-top:5px;">Green: AIS Active | Red: High Risk | Dark Gaps are empty.</p>
    </div>
  </div>

  <script>
    const p = {payload_json};
    document.getElementById('topMmsi').innerText = p.top_mmsi;

    const map = L.map('map').setView(p.center, 3);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '&copy; CartoDB' }}).addTo(map);

    // Zones
    for (const z of p.zones) {{
      L.circle([z.lat, z.lon], {{ radius: z.radius_km * 1000, color: z.color, weight: 1, fillOpacity: 0.1 }}).addTo(map).bindPopup(z.name);
    }}

    // Layers
    const overlays = {{}};
    for (const [name, markers] of Object.entries(p.layers)) {{
      const group = L.layerGroup();
      markers.forEach(m => {{
        L.circleMarker([m.lat, m.lon], {{ radius: 5, color: m.color, fillColor: m.color, fillOpacity: 0.8, weight: 1 }})
          .bindPopup(m.popup).addTo(group);
      }});
      group.addTo(map);
      overlays[name] = group;
    }}
    L.control.layers(null, overlays, {{ collapsed: true }}).addTo(map);

    // Summary Stats
    const s = p.summary;
    document.getElementById('stats').innerHTML = `
      <div class="stat-card"><span class="stat-val">${{s.hidden_intrusions}}</span><span class="stat-lbl">AI Intrusions</span></div>
      <div class="stat-card"><span class="stat-val">${{s.rendezvous}}</span><span class="stat-lbl">Rendezvous</span></div>
      <div class="stat-card"><span class="stat-val">${{s.dark}}</span><span class="stat-lbl">Dark Gaps</span></div>
      <div class="stat-card"><span class="stat-val">${{s.vessels}}</span><span class="stat-lbl">Vessels</span></div>
    `;

    // Radar Chart
    new Chart(document.getElementById('radarChart'), {{
      type: 'radar',
      data: {{
        labels: ['Dark Gaps', 'Zone Intrusion', 'High Activity', 'Rendezvous'],
        datasets: p.radar.map((d, i) => ({{
          label: d.mmsi,
          data: [d.dark, d.intrusion, d.activity, d.rendezvous],
          backgroundColor: `rgba(56, 189, 248, 0.2)`,
          borderColor: ['#38bdf8', '#fb7185', '#34d399', '#fbbf24', '#a78bfa'][i],
          borderWidth: 2
        }}))
      }},
      options: {{ 
        scales: {{ r: {{ grid: {{ color: '#334151' }}, angleLines: {{ color: '#334151' }}, ticks: {{ display: false }} }} }},
        plugins: {{ legend: {{ labels: {{ color: '#f8fafc', font: {{ size: 10 }} }} }} }}
      }}
    }});

    // Timeline Chart
    new Chart(document.getElementById('timelineChart'), {{
      type: 'bar',
      data: {{
        labels: p.timeline.map(d => ''),
        datasets: [{{
          label: 'Transmission',
          data: p.timeline.map(d => d.v),
          backgroundColor: p.timeline.map(d => d.a ? '#fb7185' : '#34d399'),
        }}]
      }},
      options: {{
        scales: {{ x: {{ display: false }}, y: {{ display: false }} }},
        plugins: {{ legend: {{ display: false }} }}
      }}
    }});
  </script>
</body>
</html>
"""
    OUTPUT.write_text(html_text, encoding="utf-8")

def main() -> None:
    df = add_features(load_data())
    model_path = MODELS_DIR / "truth_engine_bilstm.keras"
    scaler_path = MODELS_DIR / "scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        detect_hidden_intrusions(df, model, scaler)
    else: print("Warning: Truth Engine models not found.")
    rendezvous_summary, raw_clusters = detect_rendezvous(df)
    if not raw_clusters.empty:
        df = df.merge(raw_clusters[["mmsi", "timestamp", "cluster"]].rename(columns={"cluster": "rendezvous_cluster"}), on=["mmsi", "timestamp"], how="left")
        df["rendezvous_event"] = df["rendezvous_cluster"].notna()
        df.loc[df["rendezvous_event"], "risk_score"] += 40
    else: df["rendezvous_event"] = False
    payload = build_map_payload(df, rendezvous_summary)
    write_html(payload)
    print(f"Nereus map ready: {OUTPUT}")
    print(json.dumps(payload["summary"], indent=2))

if __name__ == "__main__":
    main()
