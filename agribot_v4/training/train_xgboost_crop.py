"""
training/train_xgboost_crop.py
XGBoost Crop Recommendation Model — NEW (for comparison with RandomForest baseline)

Trains on the SAME dataset and features as the baseline RandomForest so the
comparison is fair.

Saves (separate from baseline — never overwrites):
  ../models/xgb_crop_model.pkl
  ../models/xgb_crop_label_encoder.pkl
  ../models/xgb_crop_scaler.pkl
  ../models/xgb_crop_meta.json
  ../models/comparison_crop_rf_vs_xgb.json   ← slide-ready comparison table
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "Crop_recommendation.csv"))
    df = df.rename(columns={"label": "crop"})
    df["crop"] = df["crop"].str.lower().str.strip()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + ["crop"])
    return df


def compute_metrics(y_true, y_pred, label_names):
    return {
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro":   round(float(precision_score(y_true, y_pred, average="macro",  zero_division=0)), 4),
        "recall_macro":      round(float(recall_score(y_true, y_pred,    average="macro",  zero_division=0)), 4),
        "f1_macro":          round(float(f1_score(y_true, y_pred,        average="macro",  zero_division=0)), 4),
        "precision_weighted":round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted":   round(float(recall_score(y_true, y_pred,    average="weighted", zero_division=0)), 4),
        "f1_weighted":       round(float(f1_score(y_true, y_pred,        average="weighted", zero_division=0)), 4),
    }


def train():
    print("=" * 60)
    print("  XGBoost Crop Recommendation Model (Comparison)")
    print("=" * 60)

    # ── Load identical data as baseline ──────────────────────────────────────
    print("\n[DATA] Loading Crop_recommendation.csv ...")
    df = load_data()
    print(f"       → {len(df)} rows | {df['crop'].nunique()} crops")

    le = LabelEncoder()
    y  = le.fit_transform(df["crop"].values)
    sc = StandardScaler()
    X  = sc.fit_transform(df[FEATURES].values)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    label_names = le.classes_

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        print("[WARN] xgboost not installed. Installing ...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        from xgboost import XGBClassifier
        xgb_available = True

    print("\n[XGB] Training XGBoost ...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr)
    xgb_time = round(time.time() - t0, 2)

    xgb_pred = xgb.predict(X_te)
    xgb_metrics = compute_metrics(y_te, xgb_pred, label_names)
    print(f"[XGB] Accuracy={xgb_metrics['accuracy']:.4f}  F1={xgb_metrics['f1_macro']:.4f}  "
          f"Time={xgb_time}s")
    print(classification_report(y_te, xgb_pred, target_names=label_names, zero_division=0))

    # ── Train baseline RF for comparison on same split ───────────────────────
    print("\n[RF]  Training RandomForest baseline (same split) ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_time = round(time.time() - t0, 2)

    rf_pred = rf.predict(X_te)
    rf_metrics = compute_metrics(y_te, rf_pred, label_names)
    print(f"[RF]  Accuracy={rf_metrics['accuracy']:.4f}  F1={rf_metrics['f1_macro']:.4f}  "
          f"Time={rf_time}s")

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    print("\n[CV]  5-fold cross-validation ...")
    xgb_cv = cross_val_score(xgb, X, y, cv=5, scoring="accuracy")
    rf_cv  = cross_val_score(rf,  X, y, cv=5, scoring="accuracy")
    print(f"[CV]  XGBoost: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
    print(f"[CV]  RF:      {rf_cv.mean():.4f}  ± {rf_cv.std():.4f}")

    # ── Save XGBoost model ────────────────────────────────────────────────────
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_crop_model.pkl"))
    joblib.dump(le,  os.path.join(MODEL_DIR, "xgb_crop_label_encoder.pkl"))
    joblib.dump(sc,  os.path.join(MODEL_DIR, "xgb_crop_scaler.pkl"))

    xgb_meta = {
        "model":    "XGBoost",
        "features": FEATURES,
        "classes":  list(le.classes_),
        "n_crops":  int(len(le.classes_)),
        "n_train":  len(df),
        "accuracy": xgb_metrics["accuracy"],
        "f1_macro": xgb_metrics["f1_macro"],
        "cv_mean":  round(float(xgb_cv.mean()), 4),
        "cv_std":   round(float(xgb_cv.std()),  4),
        "train_time_sec": xgb_time,
        "hyperparams": {
            "n_estimators": 300, "max_depth": 6,
            "learning_rate": 0.1, "subsample": 0.8,
        },
    }
    with open(os.path.join(MODEL_DIR, "xgb_crop_meta.json"), "w") as f:
        json.dump(xgb_meta, f, indent=2)

    # ── Comparison table (slide-ready) ────────────────────────────────────────
    winner = "XGBoost" if xgb_metrics["accuracy"] >= rf_metrics["accuracy"] else "RandomForest"
    comparison = {
        "task": "Crop Recommendation Classification",
        "dataset": "Kaggle Crop_recommendation.csv (2200 rows, 22 crops)",
        "features": FEATURES,
        "test_split": "80/20 stratified",
        "winner": winner,
        "models": {
            "RandomForest": {
                **rf_metrics,
                "cv_mean":        round(float(rf_cv.mean()), 4),
                "cv_std":         round(float(rf_cv.std()),  4),
                "train_time_sec": rf_time,
                "n_estimators":   200,
                "notes": "Baseline model — used in production chatbot",
            },
            "XGBoost": {
                **xgb_metrics,
                "cv_mean":        round(float(xgb_cv.mean()), 4),
                "cv_std":         round(float(xgb_cv.std()),  4),
                "train_time_sec": xgb_time,
                "n_estimators":   300,
                "notes": "Gradient boosting — compared against RF baseline",
            },
        },
        "slide_table": [
            ["Metric", "RandomForest (Baseline)", "XGBoost", "Winner"],
            ["Accuracy",           f"{rf_metrics['accuracy']:.4f}",
                                   f"{xgb_metrics['accuracy']:.4f}",
                                   "RF" if rf_metrics['accuracy'] > xgb_metrics['accuracy'] else "XGB"],
            ["Precision (macro)",  f"{rf_metrics['precision_macro']:.4f}",
                                   f"{xgb_metrics['precision_macro']:.4f}", "—"],
            ["Recall (macro)",     f"{rf_metrics['recall_macro']:.4f}",
                                   f"{xgb_metrics['recall_macro']:.4f}", "—"],
            ["F1-Score (macro)",   f"{rf_metrics['f1_macro']:.4f}",
                                   f"{xgb_metrics['f1_macro']:.4f}",
                                   "RF" if rf_metrics['f1_macro'] > xgb_metrics['f1_macro'] else "XGB"],
            ["CV Mean (5-fold)",   f"{rf_cv.mean():.4f}",
                                   f"{xgb_cv.mean():.4f}", "—"],
            ["Train Time (s)",     f"{rf_time}",
                                   f"{xgb_time}", "—"],
        ],
    }
    comp_path = os.path.join(MODEL_DIR, "comparison_crop_rf_vs_xgb.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Print slide-ready summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CROP MODEL COMPARISON — RF vs XGBoost")
    print("=" * 60)
    print(f"{'Metric':<22} {'RandomForest':>14} {'XGBoost':>10} {'Winner':>10}")
    print("-" * 60)
    for row in comparison["slide_table"][1:]:
        print(f"{row[0]:<22} {row[1]:>14} {row[2]:>10} {row[3]:>10}")
    print(f"\n🏆 Overall Winner: {winner}")
    print(f"Comparison saved → {comp_path}")

    return xgb_metrics, rf_metrics


if __name__ == "__main__":
    train()
