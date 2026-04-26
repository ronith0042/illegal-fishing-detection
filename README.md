# Project Nereus: Forensic AI for IUU Fishing Detection 🌊🚤

## Overview

**Project Nereus** is an intelligent AI-driven framework designed to detect **Illegal, Unreported, and Unregulated (IUU)** fishing activities using maritime AIS data. Unlike traditional rule-based systems, Nereus employs deep learning and spatiotemporal clustering to identify "intentional intent" behind suspicious vessel movements.

This project implements the full multi-modal pipeline described in the Project Nereus research paper.

---

## Core AI Modules

### 🧠 1. The Truth Engine (Bi-LSTM)
*   **Purpose:** High-fidelity trajectory reconstruction during "dark periods" (AIS signal gaps).
*   **Technology:** Bidirectional Long Short-Term Memory (Bi-LSTM) Neural Network.
*   **Function:** When a vessel disables its AIS to hide its activity, the Truth Engine predicts its most likely path. If this path crosses into a protected Marine Protected Area (MPA), it flags a **Hidden Zone Intrusion** (+100 risk points).

### 🤝 2. The Handshake Detector (DBSCAN)
*   **Purpose:** Identifying clandestine transshipments (cargo transfers at sea).
*   **Technology:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
*   **Function:** Detects clusters of 2+ vessels operating at <2 knots within 500 meters of each other for more than 2 hours in offshore locations.

### ⚖️ 3. Geo-Spatial Intent Scoring
*   **Purpose:** Ranking vessels by forensic risk.
*   **Weights:**
    *   **Zone Intrusion (Real or AI-Predicted):** +100 Points
    *   **Tactical Darkness (Signal cut near borders):** +50 Points
    *   **Rendezvous Cluster Flag:** +40 Points
    *   **GPS Spoofing/Jumps:** +35 Points

---

## 📊 Forensic Dashboard

The system generates a high-end interactive dashboard (`NEREUS_ANOMALY_MAP.html`) featuring:

*   **Global Anomaly Map:** Multi-layered visualization of AI intrusions, tactical darkness, and loitering.
*   **Forensic Radar Chart:** A "Spider Chart" comparing the top 5 suspects across four risk axes.
*   **Transmission Timeline:** A "barcode" visualization showing AIS active vs. dark periods for the #1 suspect.
*   **Live Intel Panel:** Real-time summary of total AI detections and fleet activity.

---

## Project Structure

```text
illegal-fishing-detection/
│
├── data/                  # AIS CSV datasets (Trollers, Pole-and-Line)
├── models/                # Trained Bi-LSTM weights and Scalers
├── src/
│   ├── nereus_map.py      # Main application and forensic pipeline
│   └── truth_engine.py    # AI training and sequence generation logic
│
├── NEREUS_ANOMALY_MAP.html # Generated forensic dashboard
├── README.md
└── LICENSE
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install pandas numpy tensorflow scikit-learn joblib tqdm
```

### 2. Train the AI (Optional)
If you wish to retrain the Truth Engine on your local data:
```bash
python src/truth_engine.py
```
*This will generate `models/truth_engine_bilstm.keras`.*

### 3. Generate Forensic Report
```bash
python src/nereus_map.py
```
*This processes the data, runs the Bi-LSTM reconstruction, and outputs the HTML dashboard.*

---

## Technologies Used
*   **Deep Learning:** TensorFlow/Keras (Bi-LSTM)
*   **Machine Learning:** Scikit-Learn (DBSCAN)
*   **Data Science:** Pandas, NumPy
*   **Visualization:** Leaflet.js, Chart.js

---

## Author
Implementation of the **Project Nereus** AI Framework.
*(Based on Jaypee Institute of Information Technology Research)*

---

## License
MIT License
