# Illegal Fishing Detection System 🚢

## Overview

This project detects suspicious maritime behavior using AIS (Automatic Identification System) data. It identifies anomalies such as AIS signal loss, GPS spoofing, loitering, and vessel rendezvous, and visualizes them on an interactive map.

The system processes large-scale vessel tracking data and converts it into meaningful insights for monitoring illegal fishing activities.

---

## Features

### 🔍 Anomaly Detection

* **AIS Gaps (Dark Activity):** Long signal loss periods
* **GPS Jump / Spoofing:** Unrealistic movement speeds
* **Loitering:** Low-speed repeated movement offshore
* **Fishing Risk:** Fishing near protected or coastal zones
* **Rendezvous Detection:** Multiple vessels clustering offshore

---

### 📊 Risk Scoring

Each AIS point is assigned a **risk score** based on:

* signal gaps
* suspicious movement
* proximity to risk zones
* fishing activity

---

### 🗺️ Interactive Map

* Layer-based toggle system
* Clickable anomaly points with explanations
* Vessel trajectory visualization
* Risk zones (geofenced areas)

---

## Project Structure

```text
illegal_fishing_project/
│
├── data/
│   ├── trollers.csv
│   ├── pole_and_line.csv
│
├── src/
│   ├── nereus_map.py
│   ├── inspect_trollers.py
│
├── notebooks/
├── NEREUS_ANOMALY_MAP.html
├── README.md
├── LICENSE
├── .gitignore
```

---

## How It Works

1. **Load Data**

   * Reads AIS datasets from `/data`

2. **Feature Engineering**

   * Time gaps between signals
   * Distance traveled (Haversine)
   * Speed estimation
   * Course changes
   * Distance to risk zones

3. **Anomaly Detection**

   * Dark activity
   * GPS spoofing
   * Loitering
   * Fishing in sensitive zones

4. **Rendezvous Detection**

   * Identifies clusters of vessels operating together offshore

5. **Visualization**

   * Generates interactive Leaflet map
   * Saves output as:

     ```
     NEREUS_ANOMALY_MAP.html
     ```

---

## How to Run

### Install dependencies

```bash
pip install pandas numpy
```

### Run the project

```bash
python src/nereus_map.py
```

Or using entry script:

```python
from nereus_map import main

if __name__ == "__main__":
    main()
```

---

## Output

* `NEREUS_ANOMALY_MAP.html`

  * Interactive anomaly visualization
  * Toggle different anomaly layers
  * Click pins to see detailed explanations

---

## Technologies Used

* Python
* Pandas & NumPy
* Geospatial calculations (Haversine)
* Leaflet.js (map rendering)

---

## Use Case

This system can be used for:

* Maritime surveillance
* Illegal fishing detection
* Vessel behavior analysis
* Ocean monitoring systems

---

## Future Improvements

* Real-time AIS streaming
* Deep learning (Bi-LSTM across vessels)
* Dashboard (Streamlit / Web App)
* Satellite data integration

---

## Author

Ronit
(Illegal Fishing Detection Project)

---

## License

MIT License
