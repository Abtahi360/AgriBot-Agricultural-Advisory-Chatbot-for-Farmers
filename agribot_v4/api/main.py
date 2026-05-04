"""
api/main.py  — AgriBot FastAPI Backend v2
Smarter parameter extraction — handles natural language, not just exact numeric input.
"""

from __future__ import annotations
import re, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

import api.predictor as predictor
import api.chatbot   as chatbot
from api.reasoning_engine import reasoning_engine
from api.weather_api import weather_api

app = FastAPI(
    title="AgriBot — Agricultural Advisory Chatbot",
    description="NLP-powered chatbot for crop recommendation, fertilizer dosing, disease diagnosis, and weather planting advice.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────
class CropInput(BaseModel):
    N: float = Field(..., example=90)
    P: float = Field(..., example=42)
    K: float = Field(..., example=43)
    temperature: float = Field(..., example=25.0)
    humidity: float = Field(..., example=80.0)
    ph: float = Field(..., example=6.5)
    rainfall: float = Field(..., example=200.0)
    top_k: int = Field(3, ge=1, le=10)

class FertilizerInput(BaseModel):
    temperature: float = Field(..., example=32.0)
    humidity: float = Field(..., example=51.0)
    moisture: float = Field(..., example=41.0)
    soil_type: str = Field(..., example="Sandy")
    crop_type: str = Field(..., example="Wheat")
    nitrogen: float = Field(..., example=10.0)
    potassium: float = Field(..., example=5.0)
    phosphorous: float = Field(..., example=15.0)
    top_k: int = Field(3, ge=1, le=10)

class ChatMessage(BaseModel):
    message: str = Field(..., example="My rice leaves are turning yellow. What disease is this?")

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    data: Optional[dict] = None


# ── Smart parameter extraction ────────────────────────────────────────────────
def _ef(text: str, *keys) -> Optional[float]:
    for k in keys:
        m = re.search(rf"(?i)\b{k}\s*[=:is]?\s*([\d.]+)", text)
        if m:
            try: return float(m.group(1))
            except: pass
    return None


# Informal → numeric weather mapping
INFORMAL_TEMP = {
    "very hot": 38, "hot": 33, "warm": 28, "mild": 24,
    "cool": 18, "cold": 12, "very cold": 6, "freezing": 2,
}
INFORMAL_RAIN = {
    "very heavy rain": 350, "heavy rain": 250, "moderate rain": 150,
    "little rain": 60, "very little rain": 30, "no rain": 15, "dry": 20,
    "rainy season": 200, "monsoon": 300, "flood": 400, "drought": 15,
}
INFORMAL_HUM = {
    "very humid": 90, "humid": 80, "moderate humidity": 65,
    "low humidity": 40, "dry air": 30, "very dry": 20,
}

SOIL_DEFAULTS = {
    "sandy": {"N": 25, "P": 20, "K": 20, "ph": 6.5},
    "clay":  {"N": 60, "P": 40, "K": 35, "ph": 7.0},
    "loamy": {"N": 70, "P": 45, "K": 40, "ph": 6.5},
    "red":   {"N": 30, "P": 25, "K": 25, "ph": 6.0},
    "black": {"N": 55, "P": 38, "K": 32, "ph": 7.2},
}

FERT_SOIL_TYPES = ["Red", "Black", "Sandy", "Loamy", "Clayey"]
FERT_CROP_TYPES = ["Ground Nuts","Cotton","Sugarcane","Wheat","Tobacco",
                   "Barley","Millets","Pulses","Oil seeds","Maize","Paddy"]

# Map common crop names to fertilizer dataset crop names
CROP_TO_FERT_CROP = {
    "rice": "Paddy", "paddy": "Paddy", "maize": "Maize", "corn": "Maize",
    "wheat": "Wheat", "cotton": "Cotton", "sugarcane": "Sugarcane",
    "barley": "Barley", "groundnut": "Ground Nuts", "millet": "Millets",
    "pulse": "Pulses", "oil": "Oil seeds", "tobacco": "Tobacco",
}


def _infer_temp(text: str) -> Optional[float]:
    val = _ef(text, "temp", "temperature", "°c")
    if val: return val
    tl = text.lower()
    for phrase, val in sorted(INFORMAL_TEMP.items(), key=lambda x: -len(x[0])):
        if phrase in tl: return float(val)
    return None

def _infer_rainfall(text: str) -> Optional[float]:
    val = _ef(text, "rainfall", "rain", "precipitation", "mm")
    if val: return val
    tl = text.lower()
    for phrase, val in sorted(INFORMAL_RAIN.items(), key=lambda x: -len(x[0])):
        if phrase in tl: return float(val)
    return None

def _infer_humidity(text: str) -> Optional[float]:
    val = _ef(text, "humidity", "humid", "%")
    if val: return val
    tl = text.lower()
    for phrase, val in sorted(INFORMAL_HUM.items(), key=lambda x: -len(x[0])):
        if phrase in tl: return float(val)
    return None

def _detect_soil(text: str) -> Optional[str]:
    tl = text.lower()
    for soil, kws in chatbot.SOIL_TYPE_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            return soil.title()
    return None


def _extract_crop_params(text: str) -> Optional[dict]:
    """Extract crop prediction params — works with both exact numbers and informal language."""
    N    = _ef(text, "N", "nitrogen")
    P    = _ef(text, "P", "phosphor")
    K    = _ef(text, "K", "potassium")
    temp = _infer_temp(text)
    hum  = _infer_humidity(text)
    ph   = _ef(text, "ph", "pH")
    rain = _infer_rainfall(text)

    # If soil type mentioned, use soil defaults for missing values
    soil = _detect_soil(text)
    if soil and soil.lower() in SOIL_DEFAULTS:
        defs = SOIL_DEFAULTS[soil.lower()]
        N    = N    or defs["N"]
        P    = P    or defs["P"]
        K    = K    or defs["K"]
        ph   = ph   or defs["ph"]

    # Need at least one signal to call model
    has_signal = any(v is not None for v in [N, P, K, temp, hum, ph, rain])
    if not has_signal:
        return None

    return dict(
        N=N or 50, P=P or 30, K=K or 30,
        temperature=temp or 25.0,
        humidity=hum or 70.0,
        ph=ph or 6.5,
        rainfall=rain or 150.0,
    )


def _extract_fertilizer_params(text: str, meta: dict) -> Optional[dict]:
    """Extract fertilizer params — handles informal language."""
    temp  = _infer_temp(text)
    hum   = _infer_humidity(text)
    moist = _ef(text, "moisture")
    N     = _ef(text, "N", "nitrogen")
    K     = _ef(text, "K", "potassium")
    P     = _ef(text, "P", "phosphor")

    # Soil type
    soil = None
    tl   = text.lower()
    for s in FERT_SOIL_TYPES:
        if s.lower() in tl:
            soil = s; break
    if not soil:
        raw_soil = _detect_soil(text)
        if raw_soil:
            # Map to fertilizer dataset soil types
            mapping = {"Sandy":"Sandy","Clay":"Clayey","Loamy":"Loamy","Red":"Red","Black":"Black"}
            soil = mapping.get(raw_soil, "Loamy")

    # Crop type
    crop = None
    for c in FERT_CROP_TYPES:
        if c.lower() in tl:
            crop = c; break
    if not crop:
        for common, mapped in CROP_TO_FERT_CROP.items():
            if common in tl:
                crop = mapped; break

    # Only call if something was detected
    has_signal = any(v is not None for v in [temp, hum, moist, N, K, P]) or soil or crop
    if not has_signal:
        return None

    return dict(
        temperature=temp or 30.0,
        humidity=hum or 60.0,
        moisture=moist or 40.0,
        soil_type=soil or "Loamy",
        crop_type=crop or "Wheat",
        nitrogen=N or 15.0,
        potassium=K or 8.0,
        phosphorous=P or 12.0,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "AgriBot", "version": "2.0.0"}

@app.get("/ui", tags=["Frontend"])
def serve_ui():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/meta/crops", tags=["Meta"])
def crop_meta():
    return predictor.get_crop_meta()

@app.get("/meta/fertilizers", tags=["Meta"])
def fertilizer_meta():
    return predictor.get_fertilizer_meta()

@app.post("/predict/crop", tags=["Predictions"])
def predict_crop(body: CropInput):
    try:
        return predictor.predict_crop(
            N=body.N, P=body.P, K=body.K,
            temperature=body.temperature, humidity=body.humidity,
            ph=body.ph, rainfall=body.rainfall, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fertilizer", tags=["Predictions"])
def predict_fertilizer(body: FertilizerInput):
    try:
        return predictor.predict_fertilizer(
            temperature=body.temperature, humidity=body.humidity,
            moisture=body.moisture, soil_type=body.soil_type,
            crop_type=body.crop_type, nitrogen=body.nitrogen,
            potassium=body.potassium, phosphorous=body.phosphorous,
            top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(body: ChatMessage):
    try:
        text = body.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # 1. Classify intent
        intent_result = predictor.classify_intent(text)
        intent        = intent_result["intent"]
        confidence    = intent_result["confidence"]

        # 2. Detect state (uses real data from 1997-2020)
        state_name    = chatbot.detect_state(text)
        state_profile = predictor.get_state_profile(state_name) if state_name else None

        # 3. If state found, enrich params with real state data
        crop_params = _extract_crop_params(text)
        if state_profile and crop_params:
            # Fill in state real values where user didn't specify
            if _ef(text, "temp", "temperature", "°c") is None and not any(
                kw in text.lower() for kw in ["hot","cold","warm","cool","freezing"]):
                crop_params["temperature"] = state_profile["temperature"]
            if _ef(text, "rainfall", "rain", "mm") is None and not any(
                kw in text.lower() for kw in ["rain","dry","flood","monsoon"]):
                crop_params["rainfall"] = state_profile["rainfall"]
            if _ef(text, "humidity") is None:
                crop_params["humidity"] = state_profile["humidity"]
            if _ef(text, "N", "nitrogen") is None:
                crop_params["N"] = state_profile["N"]
            if _ef(text, "P", "phosphor") is None:
                crop_params["P"] = state_profile["P"]
            if _ef(text, "K", "potassium") is None:
                crop_params["K"] = state_profile["K"]
        elif state_profile and not crop_params:
            crop_params = {
                "N": state_profile["N"], "P": state_profile["P"], "K": state_profile["K"],
                "temperature": state_profile["temperature"], "humidity": state_profile["humidity"],
                "ph": state_profile.get("ph", 6.5), "rainfall": state_profile["rainfall"],
            }

        # 4. Call ML model based on intent
        prediction  = None
        yield_pred  = None

        if intent == "crop_recommendation":
            if crop_params:
                try:
                    prediction = predictor.predict_crop(**crop_params)
                    # Also predict yield
                    top_crop_n = crop_params.get("N", 70)
                    yield_pred = predictor.predict_yield(
                        fertilizer=65, temperature=crop_params["temperature"],
                        N=top_crop_n, P=crop_params["P"], K=crop_params["K"])
                except Exception:
                    pass

        elif intent == "fertilizer_advice":
            try:
                fert_meta = predictor.get_fertilizer_meta()
                fert_params = _extract_fertilizer_params(text, fert_meta)
                if fert_params and state_profile:
                    fert_params["temperature"] = fert_params["temperature"] or state_profile["temperature"]
                    fert_params["humidity"]    = fert_params["humidity"]    or state_profile["humidity"]
                if fert_params:
                    prediction = predictor.predict_fertilizer(**fert_params)
            except Exception:
                pass

        # 5. Reasoning engine intercept — fires BEFORE chatbot template.
        #    Handles: specific dosage questions, crop suitability with conditions.
        #    Returns None if question is outside rule scope → fall through to chatbot.
        reasoning_answer = reasoning_engine.intercept(
            text, intent,
            params=crop_params or {},
        )
        if reasoning_answer:
            return ChatResponse(
                intent=intent,
                confidence=confidence,
                response=reasoning_answer,
                data=prediction,
            )

        # 6. Generate response (chatbot template — fallback when reasoning engine passes)
        response_text = chatbot.generate_response(
            text, intent, prediction,
            state_profile=state_profile, state_name=state_name,
            yield_pred=yield_pred)

        return ChatResponse(
            intent=intent, confidence=confidence,
            response=response_text, data=prediction)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ── New routes (v3) ────────────────────────────────────────────────────────────

@app.get("/meta/states", tags=["Meta"])
def state_meta():
    """Return all Indian state soil+weather profiles learned from data."""
    return predictor.get_all_state_profiles()

@app.get("/meta/state/{state_name}", tags=["Meta"])
def get_state(state_name: str):
    profile = predictor.get_state_profile(state_name)
    if not profile:
        raise HTTPException(404, f"State '{state_name}' not found in dataset.")
    return {"state": state_name, "profile": profile}

@app.get("/meta/crop/{crop_name}", tags=["Meta"])
def get_crop_stats(crop_name: str):
    """Return dataset-learned statistics (mean/std) for a crop."""
    stats = predictor.get_crop_stats(crop_name)
    if not stats:
        raise HTTPException(404, f"Crop '{crop_name}' not found in dataset.")
    return {"crop": crop_name, "stats": stats}


class YieldInput(BaseModel):
    fertilizer:  float = Field(..., example=70.0)
    temperature: float = Field(..., example=28.0)
    N:           float = Field(..., example=75.0)
    P:           float = Field(..., example=22.0)
    K:           float = Field(..., example=18.0)

@app.post("/predict/yield", tags=["Predictions"])
def predict_yield(body: YieldInput):
    """Predict crop yield (tonnes/ha) from soil + weather inputs."""
    result = predictor.predict_yield(
        fertilizer=body.fertilizer, temperature=body.temperature,
        N=body.N, P=body.P, K=body.K)
    if not result:
        raise HTTPException(503, "Yield model not available. Run training first.")
    return result


# ── Reasoning Engine endpoints (direct access, no ML needed) ──────────────────

class DosageQuery(BaseModel):
    fertilizer: str = Field(..., example="urea")
    crop:       str = Field("default", example="rice")

class SuitabilityQuery(BaseModel):
    crop:        str            = Field(..., example="maize")
    temperature: Optional[float] = Field(None, example=30.0)
    humidity:    Optional[float] = Field(None, example=75.0)
    rainfall:    Optional[float] = Field(None, example=150.0)

@app.post("/reasoning/dosage", tags=["Reasoning"])
def get_dosage(body: DosageQuery):
    """
    Get specific fertilizer dosage for a crop.
    Example: fertilizer='urea', crop='rice' → '80-100 kg/acre with split timing'
    """
    answer = reasoning_engine.get_dosage(body.fertilizer, body.crop)
    return {"fertilizer": body.fertilizer, "crop": body.crop, "advice": answer}

@app.post("/reasoning/suitability", tags=["Reasoning"])
def check_suitability(body: SuitabilityQuery):
    """
    Check if temperature/humidity/rainfall conditions suit a crop.
    Example: crop='maize', temperature=30, humidity=75 → suitability verdict
    """
    answer = reasoning_engine.evaluate_suitability(
        body.crop, body.temperature, body.humidity, body.rainfall)
    if not answer:
        raise HTTPException(
            status_code=404,
            detail=f"Crop '{body.crop}' not in reasoning engine database. "
                   f"Supported: {reasoning_engine.supported_crops()}")
    return {"crop": body.crop, "verdict": answer}

@app.get("/reasoning/crops", tags=["Reasoning"])
def reasoning_crops():
    """List crops supported by the reasoning engine."""
    return {"crops": reasoning_engine.supported_crops()}

@app.get("/reasoning/fertilizers", tags=["Reasoning"])
def reasoning_fertilizers():
    """List fertilizers supported by the reasoning engine."""
    return {"fertilizers": reasoning_engine.supported_fertilizers()}


# ── Live Weather API routes ───────────────────────────────────────────────────

@app.get("/weather/status", tags=["Weather"])
def weather_status():
    return weather_api.status()

@app.get("/weather/current", tags=["Weather"])
def get_weather(city: str, country: Optional[str] = None):
    data = weather_api.get_current(city=city, country=country)
    if not data:
        return {"error": "Weather data unavailable. Set OPENWEATHER_API_KEY env variable.",
                "setup": "Get free key at openweathermap.org/api"}
    return data

@app.get("/weather/forecast", tags=["Weather"])
def get_forecast(city: str):
    data = weather_api.get_forecast_5day(city=city)
    if not data:
        return {"error": "Forecast unavailable. Check API key and city name."}
    return data

@app.get("/weather/planting-advice", tags=["Weather"])
def weather_planting_advice(city: str, crop: Optional[str] = None):
    return weather_api.get_planting_advice(city=city, crop=crop)


# ── Model comparison routes ────────────────────────────────────────────────────

def _load_json_safe(path: str):
    import json
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

@app.get("/compare/crop", tags=["Comparison"])
def compare_crop_models():
    d = _load_json_safe(os.path.join(os.path.dirname(__file__),
                        "..", "models", "comparison_crop_rf_vs_xgb.json"))
    return d or {"error": "Run training/train_xgboost_crop.py first"}

@app.get("/compare/nlp", tags=["Comparison"])
def compare_nlp_models():
    d = _load_json_safe(os.path.join(os.path.dirname(__file__),
                        "..", "models", "comparison_nlp_distilbert_vs_bert.json"))
    return d or {"error": "Run training/train_bert_intent.py first (GPU recommended)"}

@app.get("/compare/disease", tags=["Comparison"])
def compare_disease_models():
    d = _load_json_safe(os.path.join(os.path.dirname(__file__),
                        "..", "models", "disease", "comparison_disease_yolo_vs_efficientnet.json"))
    return d or {"error": "Run training/train_disease_*.py first"}

@app.get("/compare/all", tags=["Comparison"])
def compare_all_models():
    d = _load_json_safe(os.path.join(os.path.dirname(__file__),
                        "..", "evaluation", "final_comparison_summary.json"))
    return d or {"error": "Run evaluation/evaluate_all.py first"}
