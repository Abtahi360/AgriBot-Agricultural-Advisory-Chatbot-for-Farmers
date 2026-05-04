"""
api/predictor.py  — v3
Loads all trained models and exposes prediction functions.
New additions: yield prediction, state profile lookup.
"""

import os, json, joblib, numpy as np
from typing import Optional

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Singletons ────────────────────────────────────────────────────────────────
_crop_model = _crop_le = _crop_scaler = _crop_meta = None
_fert_model = _fert_enc = _fert_scaler = _fert_meta = None
_intent_tok = _intent_model = _intent_labels = None
_yield_model = _yield_scaler = _yield_meta = None
_state_profiles = _crop_stats = None


def _load_crop():
    global _crop_model, _crop_le, _crop_scaler, _crop_meta
    if _crop_model is None:
        _crop_model  = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
        _crop_le     = joblib.load(os.path.join(MODEL_DIR, "crop_label_encoder.pkl"))
        _crop_scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "crop_meta.json")) as f:
            _crop_meta = json.load(f)

def _load_fertilizer():
    global _fert_model, _fert_enc, _fert_scaler, _fert_meta
    if _fert_model is None:
        _fert_model  = joblib.load(os.path.join(MODEL_DIR, "fertilizer_model.pkl"))
        _fert_enc    = joblib.load(os.path.join(MODEL_DIR, "fertilizer_encoders.pkl"))
        _fert_scaler = joblib.load(os.path.join(MODEL_DIR, "fertilizer_scaler.pkl"))
        with open(os.path.join(MODEL_DIR, "fertilizer_meta.json")) as f:
            _fert_meta = json.load(f)

def _load_intent():
    global _intent_tok, _intent_model, _intent_labels
    if _intent_model is None:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        idir = os.path.join(MODEL_DIR, "intent_model")
        _intent_tok   = DistilBertTokenizerFast.from_pretrained(idir)
        _intent_model = DistilBertForSequenceClassification.from_pretrained(idir)
        _intent_model.eval()
        with open(os.path.join(idir, "intent_labels.json")) as f:
            _intent_labels = json.load(f)

def _load_yield():
    global _yield_model, _yield_scaler, _yield_meta
    if _yield_model is None:
        yp = os.path.join(MODEL_DIR, "yield_model.pkl")
        if os.path.exists(yp):
            _yield_model  = joblib.load(yp)
            _yield_scaler = joblib.load(os.path.join(MODEL_DIR, "yield_scaler.pkl"))
            with open(os.path.join(MODEL_DIR, "yield_meta.json")) as f:
                _yield_meta = json.load(f)

def _load_state_profiles():
    global _state_profiles
    if _state_profiles is None:
        sp = os.path.join(MODEL_DIR, "state_profiles.json")
        if os.path.exists(sp):
            with open(sp) as f:
                _state_profiles = json.load(f)
        else:
            _state_profiles = {}

def _load_crop_stats():
    global _crop_stats
    if _crop_stats is None:
        cs = os.path.join(MODEL_DIR, "crop_stats.json")
        if os.path.exists(cs):
            with open(cs) as f:
                _crop_stats = json.load(f)
        else:
            _crop_stats = {}


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_intent(text: str) -> dict:
    _load_intent()
    import torch
    inputs = _intent_tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = _intent_model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().tolist()
    id2label = _intent_labels["id2label"]
    best     = int(np.argmax(probs))
    return {
        "intent":     id2label[str(best)],
        "confidence": round(probs[best], 4),
        "scores":     {id2label[str(i)]: round(p, 4) for i, p in enumerate(probs)},
    }


def predict_crop(N, P, K, temperature, humidity, ph, rainfall, top_k=3) -> dict:
    _load_crop()
    X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    X = _crop_scaler.transform(X)
    if hasattr(_crop_model, "predict_proba"):
        probs   = _crop_model.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        recs    = [{"crop": _crop_le.classes_[i],
                    "probability": round(float(probs[i]), 4)} for i in top_idx]
    else:
        pred = _crop_model.predict(X)[0]
        recs = [{"crop": _crop_le.classes_[pred], "probability": 1.0}]
    return {
        "top_recommendations": recs,
        "input": {"N":N,"P":P,"K":K,"temperature":temperature,
                  "humidity":humidity,"ph":ph,"rainfall":rainfall},
    }


def predict_fertilizer(temperature, humidity, moisture, soil_type,
                        crop_type, nitrogen, potassium, phosphorous, top_k=3) -> dict:
    _load_fertilizer()

    def safe_enc(enc, val):
        classes = list(enc.classes_)
        # Exact match
        if val in classes: return enc.transform([val])[0]
        # Case-insensitive match
        val_l = val.lower()
        for c in classes:
            if c.lower() == val_l: return enc.transform([c])[0]
        # Partial match
        for c in classes:
            if val_l in c.lower() or c.lower() in val_l:
                return enc.transform([c])[0]
        return 0  # fallback

    soil_enc = safe_enc(_fert_enc["soil_type"], soil_type)
    crop_enc = safe_enc(_fert_enc["crop_type"], crop_type)

    X = np.array([[temperature, humidity, moisture,
                   soil_enc, crop_enc, nitrogen, potassium, phosphorous]])
    X = _fert_scaler.transform(X)
    probs   = _fert_model.predict_proba(X)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    le_f    = _fert_enc["fertilizer"]
    recs    = [{"fertilizer": le_f.classes_[i],
                "probability": round(float(probs[i]), 4)} for i in top_idx]
    return {
        "top_recommendations": recs,
        "available_soil_types": _fert_meta["soil_types"],
        "available_crop_types": _fert_meta["crop_types"],
        "input": {"temperature":temperature,"humidity":humidity,"moisture":moisture,
                  "soil_type":soil_type,"crop_type":crop_type,
                  "nitrogen":nitrogen,"potassium":potassium,"phosphorous":phosphorous},
    }


def predict_yield(fertilizer, temperature, N, P, K) -> Optional[dict]:
    """Predict numeric yield (kg/ha equivalent) given soil + weather."""
    _load_yield()
    if _yield_model is None:
        return None
    X = np.array([[fertilizer, temperature, N, P, K]])
    X = _yield_scaler.transform(X)
    y = float(_yield_model.predict(X)[0])
    return {"predicted_yield": round(y, 2),
            "unit": "tonnes/ha (approx)",
            "note": "Based on training data averages. Actual yield varies by variety and management."}


def get_state_profile(state_name: str) -> Optional[dict]:
    """Return soil + weather profile for an Indian state."""
    _load_state_profiles()
    if not _state_profiles: return None
    # Exact match
    if state_name in _state_profiles: return _state_profiles[state_name]
    # Case-insensitive
    sl = state_name.lower()
    for k, v in _state_profiles.items():
        if k.lower() == sl: return v
    # Partial
    for k, v in _state_profiles.items():
        if sl in k.lower() or k.lower() in sl: return v
    return None


def get_all_state_profiles() -> dict:
    _load_state_profiles()
    return _state_profiles or {}


def get_crop_stats(crop_name: str) -> Optional[dict]:
    """Return mean/std of soil+weather for a given crop (learned from data)."""
    _load_crop_stats()
    if not _crop_stats: return None
    cn = crop_name.lower().strip()
    if cn in _crop_stats: return _crop_stats[cn]
    for k, v in _crop_stats.items():
        if cn in k or k in cn: return v
    return None


def get_crop_meta() -> dict:
    _load_crop()
    return _crop_meta

def get_fertilizer_meta() -> dict:
    _load_fertilizer()
    return _fert_meta
