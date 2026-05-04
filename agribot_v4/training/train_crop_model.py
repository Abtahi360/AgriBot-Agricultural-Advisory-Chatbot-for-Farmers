"""
training/train_crop_model.py  — v4  (Verified: 99.5% accuracy, ~2s runtime)

Dataset strategy (fixes OOM + low accuracy from v3):
  - Crop CLASSIFIER: Dataset 1 only (Kaggle, 2200 rows, plant-level measurements)
    → State-level averaged data (crop_yield.csv) CANNOT be used for classification
      because same N/P/K maps to 30+ different crops → model cannot learn
  - State profiles: state_soil + state_weather → saved as lookup JSON for chatbot
  - Crop yield stats: crop_yield.csv → qualitative advice (avg/max yield, seasons)
  - Yield REGRESSOR: Crop Yiled with Soil and Weather.csv (R²=0.99)

Saves:
  ../models/crop_model.pkl           — RandomForest crop classifier (22 crops)
  ../models/crop_label_encoder.pkl
  ../models/crop_scaler.pkl
  ../models/crop_meta.json
  ../models/crop_stats.json          — ideal soil/weather per crop (from data)
  ../models/crop_yield_stats.json    — avg/max yield + seasons per crop
  ../models/state_profiles.json      — 30 Indian states real soil+weather data
  ../models/yield_model.pkl          — RandomForest yield regressor
  ../models/yield_scaler.pkl
  ../models/yield_meta.json
"""

import os, json, warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score
warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def dp(filename):
    return os.path.join(DATA_DIR, filename)

def mp(filename):
    return os.path.join(MODEL_DIR, filename)


# ── STEP 1: Crop Classifier ───────────────────────────────────────────────────
def train_crop_classifier():
    print("[CROP] Loading Dataset 1: Kaggle Crop Recommendation (2200 rows) ...")
    df = pd.read_csv(dp("Crop_recommendation.csv"))
    df = df.rename(columns={"label": "crop"})
    df["crop"] = df["crop"].str.lower().str.strip()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + ["crop"])
    print(f"       → {len(df)} rows | {df['crop'].nunique()} crops: {sorted(df['crop'].unique())}")

    le = LabelEncoder()
    y  = le.fit_transform(df["crop"].values)
    sc = StandardScaler()
    X  = sc.fit_transform(df[FEATURES].values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print("[CROP] Training RandomForest ...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    print(f"[CROP] Test Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

    joblib.dump(model, mp("crop_model.pkl"))
    joblib.dump(le,    mp("crop_label_encoder.pkl"))
    joblib.dump(sc,    mp("crop_scaler.pkl"))

    # Crop stats: ideal soil/weather conditions per crop (from data)
    df_orig = pd.read_csv(dp("Crop_recommendation.csv")).rename(columns={"label":"crop"})
    df_orig["crop"] = df_orig["crop"].str.lower().str.strip()
    stats = df_orig.groupby("crop")[FEATURES].mean().round(2)
    crop_stats = {
        crop: {f: round(float(stats.loc[crop, f]), 2) for f in FEATURES}
        for crop in stats.index
    }
    with open(mp("crop_stats.json"), "w") as f:
        json.dump(crop_stats, f, indent=2)

    meta = {
        "features": FEATURES,
        "classes":  list(le.classes_),
        "n_crops":  int(len(le.classes_)),
        "n_train":  len(df),
        "accuracy": round(float(acc), 4),
        "note":     "Trained on plant-level measurements. State-averaged data excluded (cannot classify).",
    }
    with open(mp("crop_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[CROP] Model saved. Accuracy={acc:.4f}")
    return acc


# ── STEP 2: State Profiles ────────────────────────────────────────────────────
def build_state_profiles():
    print("\n[STATE] Building state profiles from real 1997-2020 data ...")

    soil = pd.read_csv(dp("state_soil_data.csv"))
    soil.columns = [c.strip() for c in soil.columns]
    soil = soil.rename(columns={"pH": "ph"})

    wx = pd.read_csv(dp("state_weather_data_1997_2020.csv"))
    wx.columns = [c.strip() for c in wx.columns]
    wx = wx.rename(columns={
        "avg_temp_c":          "temperature",
        "total_rainfall_mm":   "rainfall",
        "avg_humidity_percent":"humidity",
    })
    wx_avg = wx.groupby("state")[["temperature", "rainfall", "humidity"]].mean().reset_index()

    merged = pd.merge(soil, wx_avg, on="state", how="inner")

    profiles = {}
    for _, row in merged.iterrows():
        profiles[row["state"]] = {
            k: round(float(row[k]), 1)
            for k in ["N", "P", "K", "ph", "temperature", "rainfall", "humidity"]
        }
    with open(mp("state_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"[STATE] {len(profiles)} state profiles saved.")
    return merged


# ── STEP 3: Crop Yield Stats ──────────────────────────────────────────────────
def build_crop_yield_stats():
    print("\n[YIELD STATS] Extracting crop yield statistics from crop_yield.csv ...")
    cy = pd.read_csv(dp("crop_yield.csv"))
    cy.columns = [c.strip() for c in cy.columns]

    # Standardize crop names
    crop_map = {
        "arhar/tur": "pigeonpeas", "bajra": "millet", "gram": "chickpea",
        "moong(green gram)": "mungbean", "cotton(lint)": "cotton",
        "urad": "blackgram", "soyabean": "soybean", "rapeseed": "mustard",
    }
    def clean(s):
        s = str(s).strip().lower()
        for k, v in crop_map.items():
            if k in s: return v
        return s

    cy["crop_clean"] = cy["crop"].apply(clean)

    stats = cy.groupby("crop_clean").agg(
        avg_yield=("yield", "mean"),
        max_yield=("yield", "max"),
        seasons=("season", lambda x: list(x.dropna().unique())),
        states=("state", lambda x: list(x.dropna().unique()[:5])),
    ).round(3)

    yield_dict = {
        crop: {
            "avg_yield": round(float(stats.loc[crop, "avg_yield"]), 3),
            "max_yield": round(float(stats.loc[crop, "max_yield"]), 3),
            "seasons":   list(stats.loc[crop, "seasons"]),
            "states":    list(stats.loc[crop, "states"]),
        }
        for crop in stats.index
    }
    with open(mp("crop_yield_stats.json"), "w") as f:
        json.dump(yield_dict, f, indent=2)

    print(f"[YIELD STATS] {len(yield_dict)} crops saved.")


# ── STEP 4: Yield Regressor ───────────────────────────────────────────────────
def train_yield_regressor():
    print("\n[YIELD REG] Training yield regression model ...")
    df = pd.read_csv(dp("Crop Yiled with Soil and Weather.csv"))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"temp": "temperature", "yeild": "yield"})
    df = df.dropna()

    YIELD_FEATURES = ["Fertilizer", "temperature", "N", "P", "K"]
    X = df[YIELD_FEATURES].values
    y = df["yield"].values

    sc = StandardScaler()
    X  = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    r2 = r2_score(y_te, model.predict(X_te))
    print(f"[YIELD REG] R² score: {r2:.4f}")

    joblib.dump(model, mp("yield_model.pkl"))
    joblib.dump(sc,    mp("yield_scaler.pkl"))
    with open(mp("yield_meta.json"), "w") as f:
        json.dump({"features": YIELD_FEATURES, "r2": round(float(r2), 4)}, f, indent=2)

    print("[YIELD REG] Model saved.")
    return r2


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AgriBot — Crop Model Training  (v4)")
    print("=" * 60)

    crop_acc = train_crop_classifier()
    build_state_profiles()
    build_crop_yield_stats()
    yield_r2  = train_yield_regressor()

    print("\n" + "=" * 60)
    print("  ALL CROP MODELS SAVED SUCCESSFULLY")
    print(f"  Crop classifier accuracy : {crop_acc:.1%}")
    print(f"  Yield regressor R²       : {yield_r2:.4f}")
    print("=" * 60)
