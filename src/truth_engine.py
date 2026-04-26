import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

# Configuration
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 11  # Use 11 points: 5 before, 1 middle (target), 5 after
FEATURE_COLS = ["lat", "lon", "speed", "course", "rot", "time_delta"]

def calculate_rot(df):
    """Calculate Rate of Turn (RoT)"""
    df = df.sort_values(["mmsi", "timestamp"])
    df["time_delta"] = df.groupby("mmsi")["timestamp"].diff()
    df["course_diff"] = df.groupby("mmsi")["course"].diff().abs()
    # Handle 360-degree wrap around
    df["course_diff"] = np.minimum(df["course_diff"], 360 - df["course_diff"])
    df["rot"] = df["course_diff"] / (df["time_delta"] / 60.0) # Degrees per minute
    df.loc[df["time_delta"] <= 0, "rot"] = 0
    return df.fillna(0)

def preprocess_data():
    print("Loading and preprocessing data...")
    files = [DATA_DIR / "trollers.csv", DATA_DIR / "pole_and_line.csv"]
    df_list = []
    for f in files:
        if f.exists():
            df_list.append(pd.read_csv(f))
    
    if not df_list:
        raise FileNotFoundError("No CSV files found in data/")
        
    df = pd.concat(df_list)
    df = df.sort_values(["mmsi", "timestamp"])
    
    # Kinematic Cleaning (Remove unrealistic jumps)
    df = df[df["speed"].between(0, 45)]
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    
    df = calculate_rot(df)
    
    # Feature Scaling
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    
    # Save scaler for use in main application
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    return df, scaler

def create_sequences(df):
    print("Generating sequences for Bi-LSTM training...")
    sequences = []
    targets = []
    
    for mmsi, group in df.groupby("mmsi"):
        if len(group) < SEQ_LEN:
            continue
            
        data = group[FEATURE_COLS].values
        times = group["timestamp"].values
        
        for i in range(len(data) - SEQ_LEN + 1):
            window = data[i : i + SEQ_LEN]
            window_times = times[i : i + SEQ_LEN]
            
            # Clean sequences (no massive gaps within the training window)
            if (window_times[-1] - window_times[0]) > 14400: 
                continue
                
            # Self-Supervised: Mask the middle point
            target = window[SEQ_LEN // 2, :2] 
            
            input_seq = window.copy()
            input_seq[SEQ_LEN // 2, :] = 0 # Mask target features
            
            sequences.append(input_seq)
            targets.append(target)
            
    return np.array(sequences), np.array(targets)

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation="relu"),
        layers.Dense(2) # Output: Lat, Lon
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    df, scaler = preprocess_data()
    X, y = create_sequences(df)
    
    if len(X) == 0:
        print("Not enough continuous data found to train the model.")
        return

    # Split data: 80% Train, 20% Val/Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    # Further split Temp into Val and Test (50/50 of the 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Dataset Split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model = build_model((SEQ_LEN, len(FEATURE_COLS)))
    
    # Callbacks to prevent overfitting and save best version
    model_path = MODELS_DIR / "truth_engine_bilstm.keras"
    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    print("Starting training with Best Model Checkpointing...")
    history = model.fit(
        X_train, y_train,
        epochs=15, # Increased epochs since EarlyStopping will handle it
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Final Evaluation on Test Set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Set Performance - MSE: {test_loss:.6f}, MAE: {test_mae:.6f}")
    print(f"Best Truth Engine saved to {model_path}")

if __name__ == "__main__":
    main()
