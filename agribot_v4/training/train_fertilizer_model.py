"""
training/train_fertilizer_model.py  — v5  (Fixed: removed broken src.* imports)

Root cause of the crash that was fixed here:
  The file had these stale lines at the very top that crashed Python
  BEFORE train() was ever defined:

      from src.data_cleaning import clean_dataframe, remove_outliers   ← BROKEN
      from src.feature_engineering import add_environment_features      ← BROKEN
      df = pd.read_csv("data/raw/fertilizer.csv")                       ← FILE MISSING

  These lines ran at import time → ModuleNotFoundError: No module named 'src'

Fix: all preprocessing is now self-contained inside helper functions.
No external src.* or preprocessing.* package imports needed.

Saves:
  ../models/fertilizer_model.pkl
  ../models/fertilizer_encoders.pkl
  ../models/fertilizer_scaler.pkl
  ../models/fertilizer_meta.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Domain knowledge (agronomic science) ──────────────────────────────────────
SOIL_TYPES = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
CROP_TYPES = ["Wheat", "Maize", "Paddy", "Cotton", "Ground Nuts", "Sugarcane",
              "Barley", "Millets", "Pulses", "Tobacco", "Oil Seeds"]

SOIL_BASE = {
    "Sandy":  {"N": (18, 32), "P": (8,  22), "K": (12, 28)},
    "Loamy":  {"N": (58, 82), "P": (38, 58), "K": (32, 52)},
    "Black":  {"N": (42, 68), "P": (28, 48), "K": (22, 42)},
    "Red":    {"N": (20, 40), "P": (12, 30), "K": (15, 35)},
    "Clayey": {"N": (48, 72), "P": (25, 48), "K": (22, 40)},
}

CROP_DEMAND = {
    "Wheat":       {"N": 120, "P": 60,  "K": 40},
    "Maize":       {"N": 150, "P": 70,  "K": 50},
    "Paddy":       {"N": 100, "P": 50,  "K": 50},
    "Cotton":      {"N": 130, "P": 65,  "K": 55},
    "Ground Nuts": {"N":  25, "P": 50,  "K": 50},
    "Sugarcane":   {"N": 200, "P": 80,  "K": 100},
    "Barley":      {"N":  80, "P": 40,  "K": 30},
    "Millets":     {"N":  70, "P": 35,  "K": 30},
    "Pulses":      {"N":  20, "P": 60,  "K": 40},
    "Tobacco":     {"N": 100, "P": 50,  "K": 100},
    "Oil Seeds":   {"N":  50, "P": 50,  "K": 30},
}


# ── Self-contained preprocessing helpers ─────────────────────────────────────
# (replaces the broken "from src.data_cleaning import ..." calls)

def _clean_dataframe(df):
    """Normalize columns, drop NaN/duplicates, strip strings."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"  [clean] Dropped {before - len(df)} rows with missing values")
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        print(f"  [clean] Dropped {before - len(df)} duplicate rows")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def _remove_outliers(df, numeric_cols):
    """Clip outliers to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for each column."""
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    return df


def _add_environment_features(df):
    """Add temp_category, humidity_level, moisture_level, total_nutrients."""
    df = df.copy()

    def _tc(t):
        return ("cold" if t < 15 else "cool" if t < 20 else
                "moderate" if t < 25 else "warm" if t < 30 else "hot")

    def _hl(h):
        return ("very_low" if h < 30 else "low" if h < 50 else
                "moderate" if h < 70 else "high" if h < 85 else "very_high")

    def _ml(m):
        return "low" if m < 40 else "medium" if m < 60 else "high"

    if "temperature"  in df.columns: df["temp_category"]  = df["temperature"].apply(_tc)
    if "humidity"     in df.columns: df["humidity_level"] = df["humidity"].apply(_hl)
    if "moisture"     in df.columns: df["moisture_level"] = df["moisture"].apply(_ml)

    n = next((c for c in df.columns if c in ("nitrogen",    "n")), None)
    p = next((c for c in df.columns if c in ("phosphorous", "phosphorus", "p")), None)
    k = next((c for c in df.columns if c in ("potassium",   "k")), None)
    if n and p and k:
        df["total_nutrients"] = df[n] + df[p] + df[k]

    return df


# ── Agronomic fertilizer rule ──────────────────────────────────────────────────

def recommend_fertilizer(N_def, P_def, K_def):
    N_high, P_high, K_high = N_def > 40, P_def > 30, K_def > 25
    if N_high and not P_high and not K_high: return "Urea"
    if N_high and P_high and not K_high:     return "DAP" if N_def < 55 else "28-28"
    if N_high and P_high and K_high:         return "10-26-26" if K_def >= 50 else "14-35-14"
    if not N_high and P_high and K_high:     return "10-26-26"
    if N_high and not P_high and K_high:     return "17-17-17"
    if not N_high and P_high and not K_high: return "20-20"
    if N_high and P_high and N_def < 55:     return "DAP"
    return "17-17-17"


# ── Synthetic dataset ─────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_per_combo=200):
    np.random.seed(42)
    rows = []
    for soil in SOIL_TYPES:
        for crop in CROP_TYPES:
            for _ in range(n_per_combo):
                temp  = np.random.uniform(18, 42)
                hum   = np.random.uniform(30, 90)
                moist = np.random.uniform(25, 75)
                N_soil = np.random.uniform(*SOIL_BASE[soil]["N"])
                P_soil = np.random.uniform(*SOIL_BASE[soil]["P"])
                K_soil = np.random.uniform(*SOIL_BASE[soil]["K"])
                fert = recommend_fertilizer(
                    max(0.0, CROP_DEMAND[crop]["N"] - N_soil),
                    max(0.0, CROP_DEMAND[crop]["P"] - P_soil),
                    max(0.0, CROP_DEMAND[crop]["K"] - K_soil),
                )
                rows.append({
                    "temperature": round(temp,  1),
                    "humidity":    round(hum,   1),
                    "moisture":    round(moist, 1),
                    "soil_type":   soil,
                    "crop_type":   crop,
                    "nitrogen":    round(N_soil, 1),
                    "potassium":   round(K_soil, 1),
                    "phosphorous": round(P_soil, 1),
                    "fertilizer":  fert,
                })
    return pd.DataFrame(rows)


# ── Main training function ────────────────────────────────────────────────────

def train():
    print("[FERT] Generating agronomic rule-based synthetic dataset ...")
    df = generate_synthetic_dataset(n_per_combo=200)
    print(f"       → {len(df)} rows | {df['fertilizer'].nunique()} fertilizer classes")
    print(f"       → Distribution: {df['fertilizer'].value_counts().to_dict()}")

    print("[FERT] Step 1/5  — Cleaning data ...")
    df = _clean_dataframe(df)

    print("[FERT] Step 2/5  — Removing outliers ...")
    df = _remove_outliers(df, ["temperature", "humidity", "moisture",
                                "nitrogen", "phosphorous", "potassium"])

    print("[FERT] Step 3/5  — Engineering features ...")
    df = _add_environment_features(df)

    print("[FERT] Step 4/5  — Encoding & scaling ...")
    encoders = {}
    for col in ["soil_type", "crop_type"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    le_fert = LabelEncoder()
    df["fert_enc"] = le_fert.fit_transform(df["fertilizer"])
    encoders["fertilizer"] = le_fert

    FEATURES = ["temperature", "humidity", "moisture",
                "soil_type_enc", "crop_type_enc",
                "nitrogen", "potassium", "phosphorous", "total_nutrients"]
    FEATURES = [f for f in FEATURES if f in df.columns]   # safe guard

    X      = df[FEATURES].values
    y      = df["fert_enc"].values
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print("[FERT] Step 5/5  — Training RandomForest ...")
    model = RandomForestClassifier(
        n_estimators=150, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    print(f"\n[FERT] Test Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred,
          target_names=le_fert.classes_, zero_division=0))

    # ── Save ─────────────────────────────────────────────────────────────────
    joblib.dump(model,    os.path.join(MODEL_DIR, "fertilizer_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "fertilizer_encoders.pkl"))
    joblib.dump(scaler,   os.path.join(MODEL_DIR, "fertilizer_scaler.pkl"))

    FEAT_DISPLAY = ["temperature", "humidity", "moisture",
                    "soil_type", "crop_type",
                    "nitrogen", "potassium", "phosphorous"]
    meta = {
        "features":      FEAT_DISPLAY,
        "soil_types":    list(encoders["soil_type"].classes_),
        "crop_types":    list(encoders["crop_type"].classes_),
        "fertilizers":   list(le_fert.classes_),
        "n_train":       len(df),
        "accuracy":      round(float(acc), 4),
        "method":        "agronomic-rule-based synthetic data (N/P/K deficit logic)",
        "preprocessing": [
            "column name normalization",
            "missing value drop",
            "duplicate removal",
            "IQR outlier clipping",
            "environment feature engineering (temp/humidity/moisture categories)",
            "total_nutrients derived feature",
            "label encoding (soil_type, crop_type, fertilizer)",
            "standard scaling (all numeric features)",
        ],
    }
    with open(os.path.join(MODEL_DIR, "fertilizer_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[FERT] Model saved.  Accuracy = {acc:.4f}")
    return acc


if __name__ == "__main__":
    print("=" * 60)
    print("  AgriBot — Fertilizer Model Training  (v5)")
    print("=" * 60)
    acc = train()
    print("\n" + "=" * 60)
    print("  FERTILIZER MODEL SAVED SUCCESSFULLY")
    print(f"  Accuracy: {acc:.1%}")
    print("=" * 60)
