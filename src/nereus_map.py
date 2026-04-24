from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    (ROOT / "data" / "trollers.csv", "Troller"),
    (ROOT / "data" / "pole_and_line.csv", "Pole and line"),
]
OUTPUT = ROOT / "NEREUS_ANOMALY_MAP.html"

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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    previous = df.groupby("mmsi")[["lat", "lon", "timestamp", "course"]].shift(1)
    df["prev_lat"] = previous["lat"]
    df["prev_lon"] = previous["lon"]
    df["time_gap_hours"] = (df["timestamp"] - previous["timestamp"]).dt.total_seconds() / 3600
    df["step_km"] = haversine_km(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df["derived_speed_knots"] = df["step_km"] / df["time_gap_hours"] / 1.852
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
    score += np.where(df["inside_risk_zone"], 20, 0)
    df["risk_score"] = score
    return df


def detect_rendezvous(df: pd.DataFrame) -> pd.DataFrame:
    slow = df[
        (df["speed"] <= 2)
        & (df["distance_from_port"] > 50000)
        & (df["distance_from_shore"] > 10000)
    ].copy()
    if slow.empty:
        return pd.DataFrame()

    slow["day"] = slow["timestamp"].dt.strftime("%Y-%m-%d")
    slow["lat_bin"] = (slow["lat"] / 0.05).round()
    slow["lon_bin"] = (slow["lon"] / 0.05).round()
    clusters = (
        slow.groupby(["day", "lat_bin", "lon_bin"])
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            points=("mmsi", "size"),
            vessels=("mmsi", "nunique"),
            vessel_ids=("mmsi", lambda values: ", ".join(sorted(set(values))[:6])),
        )
        .reset_index()
    )
    clusters["duration_hours"] = (clusters["end"] - clusters["start"]).dt.total_seconds() / 3600
    clusters = clusters[(clusters["vessels"] >= 2) & (clusters["duration_hours"] >= 2)]
    return clusters.sort_values(["vessels", "duration_hours", "points"], ascending=False)


def sample_rows(df: pd.DataFrame, mask: pd.Series, limit: int) -> pd.DataFrame:
    rows = df[mask].sort_values("risk_score", ascending=False)
    if len(rows) > limit:
        rows = rows.head(limit)
    return rows


def popup_for_row(row: pd.Series) -> str:
    reasons = []
    if row["tactical_darkness"]:
        reasons.append(f"Tactical AIS darkness: {row['time_gap_hours']:.1f} h signal gap")
    elif row["dark_gap"]:
        reasons.append(f"AIS dark gap: {row['time_gap_hours']:.1f} h")
    if row["gps_jump"]:
        reasons.append(f"GPS jump/spoofing risk: implied {row['derived_speed_knots']:.1f} kn")
    if row["loitering"]:
        reasons.append(f"Low-speed loitering: {row['speed']:.1f} kn, {row['step_km']:.2f} km step")
    if row["fishing_zone_risk"]:
        reasons.append("Fishing activity near protected/coastal risk area")
    if row["inside_risk_zone"]:
        reasons.append(f"Inside {row['nearest_zone']}")

    reason_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in reasons)
    return (
        f"<b>MMSI:</b> {html.escape(str(row['mmsi']))}<br>"
        f"<b>Time:</b> {row['timestamp']}<br>"
        f"<b>Gear:</b> {html.escape(str(row['gear_type']))}<br>"
        f"<b>Risk score:</b> {int(row['risk_score'])}<br>"
        f"<b>Speed:</b> {row['speed']:.2f} kn<br>"
        f"<b>Nearest zone:</b> {html.escape(str(row['nearest_zone']))} ({row['nearest_zone_km']:.1f} km)<br>"
        f"<b>Anomalies:</b><ul>{reason_html}</ul>"
    )


def marker_payload(rows: pd.DataFrame, color: str) -> list[dict]:
    payload = []
    for _, row in rows.iterrows():
        payload.append(
            {
                "lat": round(float(row["lat"]), 6),
                "lon": round(float(row["lon"]), 6),
                "color": color,
                "score": int(row["risk_score"]),
                "popup": popup_for_row(row),
            }
        )
    return payload


def build_map_payload(df: pd.DataFrame, rendezvous: pd.DataFrame) -> dict:
    layers = {
        "Tactical AIS darkness": marker_payload(
            sample_rows(df, df["tactical_darkness"], MAX_PINS_PER_LAYER), "#7b2cbf"
        ),
        "GPS jump / spoofing": marker_payload(
            sample_rows(df, df["spoofing_risk"], MAX_PINS_PER_LAYER), "#111827"
        ),
        "Low-speed loitering": marker_payload(
            sample_rows(df, df["loitering"], MAX_PINS_PER_LAYER), "#f59e0b"
        ),
        "Fishing in risk area": marker_payload(
            sample_rows(df, df["fishing_zone_risk"], MAX_PINS_PER_LAYER), "#ef4444"
        ),
    }

    rendezvous_markers = []
    for _, row in rendezvous.head(MAX_PINS_PER_LAYER).iterrows():
        popup = (
            f"<b>Rendezvous / transshipment candidate</b><br>"
            f"<b>Date:</b> {html.escape(str(row['day']))}<br>"
            f"<b>Duration:</b> {row['duration_hours']:.1f} h<br>"
            f"<b>Vessels:</b> {int(row['vessels'])}<br>"
            f"<b>MMSI:</b> {html.escape(str(row['vessel_ids']))}<br>"
            f"<b>Points:</b> {int(row['points'])}<br>"
            f"<b>Anomaly:</b> multiple low-speed vessels clustered offshore"
        )
        rendezvous_markers.append(
            {
                "lat": round(float(row["lat"]), 6),
                "lon": round(float(row["lon"]), 6),
                "color": "#10b981",
                "score": 40,
                "popup": popup,
            }
        )
    layers["Rendezvous clusters"] = rendezvous_markers

    tracks = []
    top_vessels = (
        df.groupby("mmsi")["risk_score"].sum().sort_values(ascending=False).head(8).index.tolist()
    )
    for mmsi in top_vessels:
        vessel = df[df["mmsi"] == mmsi].sort_values("timestamp")
        if len(vessel) > MAX_TRACK_POINTS_PER_VESSEL:
            vessel = vessel.iloc[np.linspace(0, len(vessel) - 1, MAX_TRACK_POINTS_PER_VESSEL).astype(int)]
        tracks.append(
            {
                "mmsi": str(mmsi),
                "points": [[round(float(r.lat), 6), round(float(r.lon), 6)] for r in vessel.itertuples()],
            }
        )

    center = [float(df["lat"].median()), float(df["lon"].median())]
    summary = {
        "rows": int(len(df)),
        "vessels": int(df["mmsi"].nunique()),
        "dark": int(df["dark_gap"].sum()),
        "tactical_dark": int(df["tactical_darkness"].sum()),
        "gps": int(df["spoofing_risk"].sum()),
        "loiter": int(df["loitering"].sum()),
        "fishing": int(df["fishing_zone_risk"].sum()),
        "rendezvous": int(len(rendezvous)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
    return {"center": center, "layers": layers, "tracks": tracks, "zones": RISK_ZONES, "summary": summary}


def write_html(payload: dict) -> None:
    payload_json = json.dumps(payload)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Project Nereus IUU Anomaly Map</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; font-family: Arial, sans-serif; }}
    .panel {{ position: absolute; z-index: 900; top: 14px; left: 14px; max-width: 360px; background: #ffffff; padding: 14px; border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,.18); }}
    .panel h1 {{ font-size: 18px; margin: 0 0 8px; }}
    .panel p {{ margin: 5px 0; font-size: 13px; color: #374151; }}
    .stat {{ display: inline-block; margin: 3px 6px 3px 0; padding: 4px 7px; border-radius: 999px; background: #eef2ff; font-size: 12px; }}
    .leaflet-popup-content ul {{ padding-left: 18px; margin: 6px 0 0; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="panel">
    <h1>Project Nereus: IUU Anomaly Map</h1>
    <p>Toggle layers at top right. Click any pin to see the detected anomaly evidence.</p>
    <div id="summary"></div>
  </section>
  <script>
    const payload = {payload_json};
    const map = L.map('map').setView(payload.center, 3);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const overlays = {{}};
    const markerStyle = (color, score) => ({{
      radius: Math.max(5, Math.min(11, 4 + score / 20)),
      color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.78
    }});

    for (const zone of payload.zones) {{
      const layer = L.layerGroup();
      L.circle([zone.lat, zone.lon], {{
        radius: zone.radius_km * 1000,
        color: zone.color,
        fillColor: zone.color,
        fillOpacity: 0.08,
        weight: 2
      }}).bindPopup(`<b>${{zone.name}}</b><br>Risk scoring geofence`).addTo(layer);
      overlays[`Risk zone: ${{zone.name}}`] = layer;
      layer.addTo(map);
    }}

    for (const [name, markers] of Object.entries(payload.layers)) {{
      const layer = L.layerGroup();
      for (const item of markers) {{
        L.circleMarker([item.lat, item.lon], markerStyle(item.color, item.score))
          .bindPopup(item.popup, {{ maxWidth: 360 }})
          .addTo(layer);
      }}
      overlays[`${{name}} (${{markers.length}} pins)`] = layer;
      layer.addTo(map);
    }}

    const trackLayer = L.layerGroup();
    const trackColors = ['#2563eb', '#dc2626', '#059669', '#9333ea', '#0891b2', '#ea580c', '#4b5563', '#be123c'];
    payload.tracks.forEach((track, index) => {{
      if (track.points.length > 1) {{
        L.polyline(track.points, {{
          color: trackColors[index % trackColors.length],
          weight: 2,
          opacity: 0.65
        }}).bindPopup(`<b>Vessel track</b><br>MMSI: ${{track.mmsi}}`).addTo(trackLayer);
      }}
    }});
    overlays['High-risk vessel tracks'] = trackLayer;
    trackLayer.addTo(map);

    L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

    const s = payload.summary;
    document.getElementById('summary').innerHTML = `
      <p><b>Data:</b> ${{s.rows.toLocaleString()}} AIS records, ${{s.vessels}} vessels</p>
      <p><b>Range:</b> ${{s.start}} to ${{s.end}}</p>
      <span class="stat">Dark gaps: ${{s.dark}}</span>
      <span class="stat">Tactical dark: ${{s.tactical_dark}}</span>
      <span class="stat">GPS/spoof: ${{s.gps}}</span>
      <span class="stat">Loitering: ${{s.loiter}}</span>
      <span class="stat">Fishing risk: ${{s.fishing}}</span>
      <span class="stat">Rendezvous: ${{s.rendezvous}}</span>
    `;
  </script>
</body>
</html>
"""
    OUTPUT.write_text(html_text, encoding="utf-8")


def main() -> None:
    df = add_features(load_data())
    rendezvous = detect_rendezvous(df)
    payload = build_map_payload(df, rendezvous)
    write_html(payload)
    print(f"Nereus map ready: {OUTPUT}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
