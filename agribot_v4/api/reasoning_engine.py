"""
api/reasoning_engine.py  — AgriBot Reasoning Layer v1

Sits BETWEEN intent classification and chatbot response generation.
Provides condition-based reasoning that the pure ML classifier cannot do:

  Problem 1: "How much urea for rice per acre?"
    → ML classifier says fertilizer_advice → chatbot gives generic NPK list ❌
    → Reasoning engine intercepts "how much" + specific fertilizer name
       → returns exact dosage answer before chatbot template runs       ✅

  Problem 2: "30°C, 75% humidity maize"
    → ML classifier says weather_planting → chatbot gives vague advice   ❌
    → Reasoning engine evaluates temp/humidity against crop-specific
       thresholds → returns precise suitability verdict with explanation  ✅

Architecture:
  main.py  →  classify_intent()
           →  ReasoningEngine.intercept(text, intent, params)
           →  if engine returns a response: use it (skip chatbot template)
           →  else: fall through to chatbot.generate_response() as before

This file has ZERO ML dependencies — pure Python rules.
Import it safely without loading any model.
"""

import re
from typing import Optional, Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DOSAGE DATABASE
# Science-backed fertilizer dosage per crop per acre (India standard)
# Sources: ICAR recommendations + state agriculture department bulletins
# ─────────────────────────────────────────────────────────────────────────────

DOSAGE_DB: Dict[str, Dict[str, Dict]] = {
    # key: fertilizer_name → crop_name → dosage info
    "urea": {
        "rice": {
            "dose": "80–100 kg/acre",
            "timing": "split: 1/3 at transplanting + 1/3 at tillering (21 days) + 1/3 at panicle initiation",
            "note":  "Use coated urea in sandy soils to reduce leaching loss",
        },
        "wheat": {
            "dose": "65–80 kg/acre",
            "timing": "half at sowing + half at first irrigation (21 days after sowing)",
            "note":  "Do not broadcast urea on wet soil — incorporate immediately",
        },
        "maize": {
            "dose": "70–90 kg/acre",
            "timing": "1/3 at sowing + 1/3 at knee-high stage + 1/3 at tasselling",
            "note":  "Side-dress along rows — avoid contact with seeds",
        },
        "sugarcane": {
            "dose": "100–130 kg/acre",
            "timing": "3 splits: at planting, at 60 days, at 120 days",
            "note":  "High nitrogen demand — split applications reduce volatilisation",
        },
        "cotton": {
            "dose": "60–80 kg/acre",
            "timing": "1/3 basal + 1/3 at squaring + 1/3 at boll development",
            "note":  "Excess nitrogen delays boll opening — follow splits strictly",
        },
        "paddy": {
            "dose": "80–100 kg/acre",
            "timing": "split: 1/3 at transplanting + 1/3 at tillering + 1/3 at panicle initiation",
            "note":  "Same as rice recommendation — paddy and rice are the same crop",
        },
        "maize / corn": {
            "dose": "70–90 kg/acre",
            "timing": "1/3 at sowing + 1/3 at knee-high stage + 1/3 at tasselling",
            "note":  "Side-dress along rows — avoid contact with seeds",
        },
        "groundnut": {
            "dose": "8–12 kg/acre",
            "timing": "basal at sowing only — groundnut fixes its own nitrogen",
            "note":  "Starter dose only. Excess nitrogen suppresses nodule formation",
        },
        "soybean": {
            "dose": "8–12 kg/acre",
            "timing": "basal at sowing only",
            "note":  "Legume — minimal nitrogen needed. Rhizobium seed treatment preferred",
        },
        "tomato": {
            "dose": "35–50 kg/acre",
            "timing": "1/3 at transplanting + 1/3 at flowering + 1/3 at fruit set",
            "note":  "Fertigate through drip irrigation for best uptake",
        },
        "potato": {
            "dose": "50–70 kg/acre",
            "timing": "half at planting + half at earthing-up (30 days)",
            "note":  "Ridge the rows after top-dressing to reduce volatilisation",
        },
        "banana": {
            "dose": "100–120 kg/acre/year",
            "timing": "6 equal monthly doses starting at 2 months after planting",
            "note":  "High nitrogen demand — monthly fertigations preferred",
        },
        "default": {
            "dose": "40–60 kg/acre",
            "timing": "half at sowing + half at 30–40 days",
            "note":  "General guideline — confirm with local agriculture office",
        },
    },
    "dap": {
        "rice": {
            "dose": "40–50 kg/acre",
            "timing": "full dose as basal at transplanting",
            "note":  "Do not mix DAP with urea in the same application",
        },
        "wheat": {
            "dose": "50–60 kg/acre",
            "timing": "full dose as basal at sowing — incorporate into soil",
            "note":  "Starter fertilizer — critical for root establishment",
        },
        "maize": {
            "dose": "45–55 kg/acre",
            "timing": "full dose as basal at sowing",
            "note":  "Place 5 cm beside and below seed — avoid seed contact",
        },
        "cotton": {
            "dose": "40–50 kg/acre",
            "timing": "full dose as basal before sowing",
            "note":  "Improves early root growth and stand establishment",
        },
        "sugarcane": {
            "dose": "50–65 kg/acre",
            "timing": "full dose as basal at planting",
            "note":  "Place in furrow below cane sets",
        },
        "groundnut": {
            "dose": "50–60 kg/acre",
            "timing": "full dose as basal at sowing",
            "note":  "High phosphorus demand in groundnut — critical for pod filling",
        },
        "default": {
            "dose": "40–50 kg/acre",
            "timing": "full dose as basal at sowing",
            "note":  "General guideline — confirm with soil test result",
        },
    },
    "mop": {
        "rice": {
            "dose": "20–30 kg/acre",
            "timing": "half at transplanting + half at panicle initiation",
            "note":  "MOP = Muriate of Potash (0-0-60). Improves grain filling",
        },
        "potato": {
            "dose": "40–50 kg/acre",
            "timing": "half at planting + half at earthing-up",
            "note":  "High potassium demand for tuber development",
        },
        "banana": {
            "dose": "60–80 kg/acre/year",
            "timing": "monthly splits from 2 months after planting",
            "note":  "Potassium is most critical nutrient for banana yield",
        },
        "cotton": {
            "dose": "25–35 kg/acre",
            "timing": "half at sowing + half at boll development",
            "note":  "Improves fiber quality and boll retention",
        },
        "default": {
            "dose": "20–30 kg/acre",
            "timing": "half at sowing + half at 30–40 days",
            "note":  "General potassium guideline",
        },
    },
    "17-17-17": {
        "default": {
            "dose": "50–60 kg/acre",
            "timing": "basal application at sowing or transplanting",
            "note":  "Balanced NPK — use when soil test shows equal deficiency",
        },
        "maize": {
            "dose": "55–65 kg/acre",
            "timing": "basal at sowing + top-dress at knee-high stage",
            "note":  "Supplement with extra urea at tasselling for high-yield targets",
        },
        "vegetables": {
            "dose": "40–50 kg/acre",
            "timing": "basal at transplanting + side-dress at 3 weeks",
            "note":  "Follow with micronutrient spray at 30 days",
        },
    },
    "20-20": {
        "default": {
            "dose": "50–60 kg/acre",
            "timing": "basal application at sowing",
            "note":  "No potassium — supplement with MOP if K is also deficient",
        },
    },
    "28-28": {
        "default": {
            "dose": "40–50 kg/acre",
            "timing": "basal at sowing + repeat at mid-season if needed",
            "note":  "High-strength fertilizer — calibrate spreader carefully",
        },
    },
    "10-26-26": {
        "default": {
            "dose": "40–50 kg/acre",
            "timing": "at late vegetative or early grain-filling stage",
            "note":  "High P+K — apply when crop has adequate N but needs P and K",
        },
        "groundnut": {
            "dose": "50–60 kg/acre",
            "timing": "at pegging stage",
            "note":  "Supports pod development and oil content",
        },
    },
    "14-35-14": {
        "default": {
            "dose": "40–50 kg/acre",
            "timing": "at flowering stage",
            "note":  "High phosphorus — critical for fruit set and seed development",
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CROP SUITABILITY THRESHOLDS
# Temperature, humidity, rainfall ranges per crop (ICAR / FAO standards)
# ─────────────────────────────────────────────────────────────────────────────

CROP_THRESHOLDS: Dict[str, Dict] = {
    "rice": {
        "temp":     {"min": 20, "opt_lo": 24, "opt_hi": 30, "max": 38},
        "humidity": {"min": 60, "opt_lo": 70, "opt_hi": 90, "max": 100},
        "rainfall": {"min": 150, "good": 200, "max": 500},
        "notes": {
            "too_cold":  "Rice requires at least 20°C. Below that, germination fails and growth stalls.",
            "too_hot":   "Above 35°C, pollen sterility increases and grain set drops sharply.",
            "optimal":   "Ideal rice conditions. High probability of good yield.",
            "low_hum":   "Rice prefers humidity >60%. Low humidity stresses the crop — increase irrigation.",
            "high_hum":  "High humidity is fine for rice but increases fungal disease risk.",
        },
    },
    "maize": {
        "temp":     {"min": 10, "opt_lo": 20, "opt_hi": 30, "max": 38},
        "humidity": {"min": 45, "opt_lo": 55, "opt_hi": 75, "max": 90},
        "rainfall": {"min": 60, "good": 100, "max": 300},
        "notes": {
            "too_cold":  "Maize needs at least 10°C to germinate. Below 18°C, growth is very slow.",
            "too_hot":   "Above 35°C, tassels dry out and pollination fails — yield drops significantly.",
            "optimal":   "Excellent conditions for maize. Plan your planting now.",
            "low_hum":   "Humidity below 45% causes water stress during silking — irrigate immediately.",
            "high_hum":  "Humidity above 85% increases grey leaf spot and northern corn blight risk.",
        },
    },
    "wheat": {
        "temp":     {"min": 5, "opt_lo": 15, "opt_hi": 22, "max": 30},
        "humidity": {"min": 30, "opt_lo": 45, "opt_hi": 65, "max": 80},
        "rainfall": {"min": 30, "good": 75, "max": 200},
        "notes": {
            "too_cold":  "Wheat can survive light frost but germination needs >5°C.",
            "too_hot":   "Above 28°C during grain filling causes shrivelled grains and yield loss.",
            "optimal":   "Good conditions for wheat. Low humidity reduces rust risk.",
            "low_hum":   "Low humidity is actually good for wheat — reduces fungal disease.",
            "high_hum":  "High humidity significantly increases wheat rust and Fusarium head blight risk.",
        },
    },
    "cotton": {
        "temp":     {"min": 18, "opt_lo": 25, "opt_hi": 35, "max": 42},
        "humidity": {"min": 40, "opt_lo": 50, "opt_hi": 75, "max": 85},
        "rainfall": {"min": 50, "good": 100, "max": 200},
        "notes": {
            "too_cold":  "Cotton is tropical — below 18°C, growth stops and bolls may drop.",
            "too_hot":   "Above 38°C, flower drop increases. Irrigate more frequently.",
            "optimal":   "Good conditions for cotton growth and boll development.",
            "low_hum":   "Cotton handles dry conditions well — good for fiber quality.",
            "high_hum":  "High humidity promotes boll rot and bacterial blight. Ensure good air flow.",
        },
    },
    "sugarcane": {
        "temp":     {"min": 20, "opt_lo": 25, "opt_hi": 35, "max": 42},
        "humidity": {"min": 50, "opt_lo": 60, "opt_hi": 85, "max": 95},
        "rainfall": {"min": 100, "good": 200, "max": 600},
        "notes": {
            "too_cold":  "Sugarcane needs tropical warmth — below 20°C growth slows significantly.",
            "too_hot":   "Tolerates high heat well with adequate irrigation.",
            "optimal":   "Ideal conditions for sugarcane tillering and cane growth.",
            "low_hum":   "Maintain irrigation frequency — sugarcane needs consistent moisture.",
            "high_hum":  "Very high humidity with heat can promote red rot — monitor regularly.",
        },
    },
    "banana": {
        "temp":     {"min": 15, "opt_lo": 25, "opt_hi": 35, "max": 40},
        "humidity": {"min": 50, "opt_lo": 70, "opt_hi": 90, "max": 100},
        "rainfall": {"min": 80, "good": 120, "max": 400},
        "notes": {
            "too_cold":  "Banana is highly frost-sensitive. Chilling below 14°C causes leaf damage.",
            "too_hot":   "Above 38°C, leaf scorching occurs — provide shade if possible.",
            "optimal":   "Excellent conditions for banana growth.",
            "low_hum":   "Banana prefers high humidity — low humidity stresses fruit development.",
            "high_hum":  "High humidity with warmth is ideal for banana.",
        },
    },
    "potato": {
        "temp":     {"min": 7, "opt_lo": 15, "opt_hi": 22, "max": 28},
        "humidity": {"min": 40, "opt_lo": 55, "opt_hi": 75, "max": 85},
        "rainfall": {"min": 40, "good": 80, "max": 200},
        "notes": {
            "too_cold":  "Below 7°C, tuber formation stops. Frost kills the plant.",
            "too_hot":   "Above 25°C, tuber initiation stops completely — poor yield expected.",
            "optimal":   "Ideal cool conditions for potato tuber development.",
            "low_hum":   "Potato needs consistent moisture — irrigate every 7–10 days.",
            "high_hum":  "High humidity significantly increases late blight (Phytophthora) risk.",
        },
    },
    "tomato": {
        "temp":     {"min": 10, "opt_lo": 20, "opt_hi": 27, "max": 35},
        "humidity": {"min": 40, "opt_lo": 50, "opt_hi": 70, "max": 80},
        "rainfall": {"min": 40, "good": 80, "max": 150},
        "notes": {
            "too_cold":  "Below 10°C, tomato flowers drop and fruit set fails.",
            "too_hot":   "Above 32°C, blossom drop occurs. Provide shade cloth or drip irrigation.",
            "optimal":   "Good conditions for tomato fruit set and development.",
            "low_hum":   "Low humidity is fine — reduces fungal disease.",
            "high_hum":  "High humidity promotes early blight and Septoria leaf spot.",
        },
    },
    "chickpea": {
        "temp":     {"min": 10, "opt_lo": 18, "opt_hi": 26, "max": 35},
        "humidity": {"min": 15, "opt_lo": 25, "opt_hi": 55, "max": 70},
        "rainfall": {"min": 30, "good": 60, "max": 150},
        "notes": {
            "too_cold":  "Chickpea can handle light frost but germination needs >10°C.",
            "too_hot":   "Above 33°C, pod set is affected — plant in cooler months.",
            "optimal":   "Good dry conditions for chickpea — it is drought tolerant.",
            "low_hum":   "Low humidity is ideal for chickpea — high humidity causes diseases.",
            "high_hum":  "High humidity strongly promotes Botrytis grey mold — avoid planting.",
        },
    },
    "groundnut": {
        "temp":     {"min": 20, "opt_lo": 25, "opt_hi": 35, "max": 40},
        "humidity": {"min": 40, "opt_lo": 50, "opt_hi": 75, "max": 85},
        "rainfall": {"min": 50, "good": 100, "max": 250},
        "notes": {
            "too_cold":  "Groundnut is warm-season — below 20°C, pod development fails.",
            "too_hot":   "Tolerates heat well but needs irrigation above 38°C.",
            "optimal":   "Good conditions for groundnut peg and pod development.",
            "low_hum":   "Moderate humidity is ideal for groundnut oil quality.",
            "high_hum":  "High humidity increases Cercospora leaf spot and aflatoxin risk.",
        },
    },
    "millet": {
        "temp":     {"min": 20, "opt_lo": 27, "opt_hi": 35, "max": 42},
        "humidity": {"min": 20, "opt_lo": 35, "opt_hi": 65, "max": 80},
        "rainfall": {"min": 25, "good": 60, "max": 200},
        "notes": {
            "too_cold":  "Millet is heat-loving — very slow growth below 20°C.",
            "too_hot":   "Highly heat tolerant. One of the best crops for hot dry climates.",
            "optimal":   "Very good conditions for millet — it thrives in warm dry weather.",
            "low_hum":   "Drought tolerant crop — low humidity is manageable.",
            "high_hum":  "High humidity can cause downy mildew — ensure good drainage.",
        },
    },
    "sorghum": {
        "temp":     {"min": 15, "opt_lo": 25, "opt_hi": 35, "max": 40},
        "humidity": {"min": 20, "opt_lo": 40, "opt_hi": 70, "max": 85},
        "rainfall": {"min": 30, "good": 75, "max": 300},
        "notes": {
            "too_cold":  "Sorghum needs warmth — slow germination below 15°C.",
            "too_hot":   "Highly heat tolerant. Better than maize in hot dry conditions.",
            "optimal":   "Good conditions for sorghum.",
            "low_hum":   "Drought tolerant — handles low humidity and dry spells well.",
            "high_hum":  "High humidity with heat may cause grain mold at harvest.",
        },
    },
}

# Aliases to map common names to threshold keys
CROP_ALIASES = {
    "paddy": "rice",
    "corn":  "maize",
    "bajra": "millet",
    "jowar": "sorghum",
    "groundnuts": "groundnut",
    "arhar": "chickpea",
    "tur":   "chickpea",
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TEXT PATTERN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Quantity-question patterns — these signal the user wants a specific amount
QTY_PATTERNS = [
    r"\bhow\s+much\b",
    r"\bhow\s+many\b",
    r"\bwhat\s+(?:is\s+the\s+)?(?:dosage|dose|amount|quantity|rate)\b",
    r"\bhow\s+(?:many|much)\s+(?:kg|kilogram|bags?|sacks?)\b",
    r"\bkg\s+per\s+(?:acre|bigha|hectare)\b",
    r"\bper\s+(?:acre|bigha|hectare)\b",
    r"\bdosage\b",
    r"\bdose\b",
]

# Fertilizer name patterns
FERT_PATTERNS = {
    "urea":     [r"\burea\b"],
    "dap":      [r"\bdap\b", r"\bdi.?ammonium\b", r"\bdi\s+ammonium\b"],
    "mop":      [r"\bmop\b", r"\bmuriate\b", r"\bpotash\b", r"\bpotassium\s+chloride\b"],
    "17-17-17": [r"\b17.17.17\b", r"\bnpk\s*17\b"],
    "20-20":    [r"\b20.20\b"],
    "28-28":    [r"\b28.28\b"],
    "10-26-26": [r"\b10.26.26\b"],
    "14-35-14": [r"\b14.35.14\b"],
}

# Crop name patterns
CROP_PATTERNS = {
    "rice":      [r"\brice\b", r"\bpaddy\b"],
    "wheat":     [r"\bwheat\b"],
    "maize":     [r"\bmaize\b", r"\bcorn\b"],
    "cotton":    [r"\bcotton\b"],
    "sugarcane": [r"\bsugarcane\b", r"\bsugar\s+cane\b"],
    "banana":    [r"\bbanana\b"],
    "potato":    [r"\bpotato\b"],
    "tomato":    [r"\btomato\b"],
    "chickpea":  [r"\bchickpea\b", r"\bchick\s+pea\b", r"\bgram\b", r"\barhar\b", r"\btur\b"],
    "groundnut": [r"\bgroundnut\b", r"\bpeanut\b", r"\barachis\b"],
    "millet":    [r"\bmillet\b", r"\bbajra\b"],
    "sorghum":   [r"\bsorghum\b", r"\bjowar\b"],
}


def _extract_float(text: str, *keywords) -> Optional[float]:
    """
    Extract a float value from text given hint keywords.
    Strategy:
      1. Keyword-anchored: "temperature 30" / "temp=30" / "humid 75"
      2. Bare degree sign fallback: "30°C"  (for temp keywords)
      3. Bare percent fallback: "75%"        (for humidity keywords)
    """
    for kw in keywords:
        m = re.search(rf"(?i)\b{re.escape(kw)}\s*[=:°]?\s*([\d.]+)", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Fallback 1 — bare NNN°C  (covers "30°C and 75% humidity")
    if any(k in ("temp", "temperature") for k in keywords):
        m = re.search(r"(\d+\.?\d*)\s*°\s*[cC]", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Fallback 2 — bare NNN%   (covers "75% humidity")
    if any(k in ("humidity", "humid") for k in keywords):
        m = re.search(r"(\d+\.?\d*)\s*%", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    return None


def _detect_fertilizer(text: str) -> Optional[str]:
    tl = text.lower()
    for name, patterns in FERT_PATTERNS.items():
        if any(re.search(p, tl) for p in patterns):
            return name
    return None


def _detect_crop(text: str) -> Optional[str]:
    tl = text.lower()
    for crop, patterns in CROP_PATTERNS.items():
        if any(re.search(p, tl) for p in patterns):
            return crop
    return None


def _is_dosage_question(text: str) -> bool:
    """Return True if user is asking for a specific quantity/amount."""
    tl = text.lower()
    return any(re.search(p, tl) for p in QTY_PATTERNS)


def _is_suitability_question(text: str) -> bool:
    """Return True if user is asking whether conditions suit a crop."""
    tl = text.lower()
    suitability_signals = [
        r"\bcan\s+i\s+(?:grow|plant|sow|cultivate)\b",
        r"\bshould\s+i\s+(?:plant|grow|sow)\b",
        r"\bis\s+(?:it\s+)?(?:good|suitable|okay|ok|safe|right|possible)\s+(?:to\s+)?(?:plant|grow|sow)\b",
        r"\bwill\s+(?:it\s+|my\s+\w+\s+)?(?:grow|survive|thrive)\b",
        r"\bsuitable\s+for\b",
        r"\bgood\s+(?:time\s+)?(?:for|to)",
        r"\b(?:too\s+)?hot\b",
        r"\b(?:too\s+)?cold\b",
        r"\bhumidity\b",
        r"\btemperature\b",
        r"\b\d+\s*°?\s*c\b",
        r"\b\d+\s*%\s*humidity\b",
    ]
    return any(re.search(p, tl) for p in suitability_signals)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RESPONSE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_dosage_response(fert: str, crop: str) -> str:
    """Build a specific dosage answer for a fertilizer + crop combination."""
    fert_data = DOSAGE_DB.get(fert.lower(), DOSAGE_DB.get("urea"))
    # Look up crop-specific entry, fall back to default
    crop_data = fert_data.get(crop.lower()) or fert_data.get("default", {})

    fert_display = fert.upper() if fert in ("dap", "mop") else fert.title()
    crop_display = crop.title()

    lines = [f"⚖️ **{fert_display} Dosage for {crop_display}**\n"]
    lines.append(f"**Recommended dose:** {crop_data.get('dose', 'See local guidelines')}")
    lines.append(f"**Application timing:** {crop_data.get('timing', 'As per standard practice')}")
    lines.append(f"**Important note:** {crop_data.get('note', '')}")

    # Cross-sell complementary fertilizers
    complementary = {
        "urea":  "For complete nutrition, also apply DAP at sowing and MOP if potassium is low.",
        "dap":   "Complement with Urea top-dressing during vegetative stage.",
        "mop":   "Ensure nitrogen needs are met with Urea or DAP application.",
    }
    if fert.lower() in complementary:
        lines.append(f"\n💡 **Tip:** {complementary[fert.lower()]}")

    lines.append("\n⚠️ *Actual dose may vary by soil test result, variety, and state recommendations.*")
    lines.append("   *Consult your local Krishi Vigyan Kendra (KVK) for field-specific advice.*")
    return "\n".join(lines)


def _build_suitability_response(crop: str, temp: Optional[float],
                                 humidity: Optional[float],
                                 rainfall: Optional[float]) -> Optional[str]:
    """
    Evaluate whether given conditions suit the crop.
    Returns None if crop not in threshold DB (fall through to chatbot).
    """
    crop_key = CROP_ALIASES.get(crop.lower(), crop.lower())
    thresholds = CROP_THRESHOLDS.get(crop_key)
    if not thresholds:
        return None  # unknown crop → let chatbot handle it

    notes   = thresholds["notes"]
    issues  = []
    goods   = []

    # ── Temperature verdict ──────────────────────────────────────────────────
    temp_verdict = "unknown"
    if temp is not None:
        t = thresholds["temp"]
        if temp < t["min"]:
            issues.append(f"❄️ **Temperature {temp}°C is too cold for {crop.title()}.**\n"
                          f"   → {notes['too_cold']}\n"
                          f"   → Minimum required: **{t['min']}°C**")
            temp_verdict = "too_cold"
        elif temp >= t["max"]:
            issues.append(f"🌡️ **Temperature {temp}°C is too hot for {crop.title()}.**\n"
                          f"   → {notes['too_hot']}\n"
                          f"   → Maximum tolerated: **{t['max']}°C**")
            temp_verdict = "too_hot"
        elif t["opt_lo"] <= temp <= t["opt_hi"]:
            goods.append(f"✅ Temperature {temp}°C — **ideal range** ({t['opt_lo']}–{t['opt_hi']}°C)")
            temp_verdict = "optimal"
        else:
            goods.append(f"⚠️ Temperature {temp}°C — **acceptable but not optimal** "
                         f"(ideal: {t['opt_lo']}–{t['opt_hi']}°C)")
            temp_verdict = "marginal"

    # ── Humidity verdict ─────────────────────────────────────────────────────
    if humidity is not None:
        h = thresholds["humidity"]
        if humidity < h["min"]:
            issues.append(f"💧 **Humidity {humidity}% is too low for {crop.title()}.**\n"
                          f"   → {notes['low_hum']}\n"
                          f"   → Minimum recommended: **{h['min']}%**")
        elif humidity > h["max"]:
            issues.append(f"💦 **Humidity {humidity}% is very high.**\n"
                          f"   → {notes['high_hum']}")
        elif h["opt_lo"] <= humidity <= h["opt_hi"]:
            goods.append(f"✅ Humidity {humidity}% — **ideal range** ({h['opt_lo']}–{h['opt_hi']}%)")
        else:
            goods.append(f"⚠️ Humidity {humidity}% — acceptable (ideal: {h['opt_lo']}–{h['opt_hi']}%)")

    # ── Rainfall verdict ─────────────────────────────────────────────────────
    if rainfall is not None:
        r = thresholds["rainfall"]
        if rainfall < r["min"]:
            issues.append(f"☀️ **Rainfall {rainfall}mm is insufficient for {crop.title()}.**\n"
                          f"   → Irrigation is essential — minimum {r['min']}mm per season needed.")
        elif rainfall > r["max"]:
            issues.append(f"🌧️ **Rainfall {rainfall}mm is too high — waterlogging risk.**\n"
                          f"   → Raise field beds. Ensure drainage channels before planting.")
        else:
            goods.append(f"✅ Rainfall {rainfall}mm — within acceptable range ({r['min']}–{r['max']}mm)")

    # ── Nothing was extracted ────────────────────────────────────────────────
    if not issues and not goods:
        return None   # no numeric data → let chatbot handle it

    # ── Build final response ─────────────────────────────────────────────────
    crop_display = crop.title()
    lines = [f"🌱 **Crop Suitability Analysis: {crop_display}**\n"]

    if issues:
        lines.append("**⚠️ Conditions that are problematic:**")
        for issue in issues:
            lines.append(f"{issue}")
        lines.append("")

    if goods:
        lines.append("**✅ Conditions that are suitable:**")
        for g in goods:
            lines.append(f"   {g}")
        lines.append("")

    # Overall verdict
    if not issues:
        lines.append(f"🟢 **Overall: Good conditions for {crop_display}.**\n"
                     f"   → {notes.get('optimal', 'Proceed with planting.')}")
    elif temp_verdict in ("too_cold", "too_hot"):
        lines.append(f"🔴 **Overall: Not recommended to plant {crop_display} now.**")
        lines.append(f"   → Address the temperature issue first before planting.")
    else:
        lines.append(f"🟡 **Overall: Marginal conditions for {crop_display}.**")
        lines.append(f"   → You can plant but take precautions for the issues listed above.")

    # Ideal conditions reference
    t = thresholds["temp"]
    h = thresholds["humidity"]
    lines.append(f"\n📊 **Ideal conditions for {crop_display}:**")
    lines.append(f"   • Temperature: {t['opt_lo']}–{t['opt_hi']}°C")
    lines.append(f"   • Humidity: {h['opt_lo']}–{h['opt_hi']}%")
    lines.append(f"   • Seasonal rainfall: {thresholds['rainfall']['good']}mm+")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN REASONING ENGINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningEngine:
    """
    Rule-based reasoning layer that intercepts requests requiring
    specific, condition-dependent answers the ML classifier cannot provide.

    Usage in main.py:
        engine   = ReasoningEngine()
        answer   = engine.intercept(text, intent, params)
        if answer:
            return answer            # skip chatbot template
        else:
            return chatbot.generate_response(...)   # normal flow
    """

    def intercept(
        self,
        text:   str,
        intent: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Try to produce a precise answer via rules.
        Returns None if the question is not in the engine's scope
        (caller should fall through to chatbot.generate_response).

        Args:
            text:   Raw user message
            intent: Intent classified by DistilBERT
            params: Extracted parameters dict (may contain temp, humidity, etc.)

        Returns:
            Formatted answer string, or None
        """
        tl = text.lower()

        # ── Rule 1: Specific dosage questions ─────────────────────────────────
        if intent == "fertilizer_advice" and _is_dosage_question(text):
            fert = _detect_fertilizer(text)
            crop = _detect_crop(text)
            if fert:
                return _build_dosage_response(fert, crop or "default")

        # ── Rule 2: Crop suitability with numeric conditions ───────────────────
        if intent in ("weather_planting", "crop_recommendation"):
            crop = _detect_crop(text)
            if crop:
                # Pull numbers from text OR from extracted params dict
                temp     = (_extract_float(text, "temp", "temperature", r"\d+\s*°c") or
                            (params or {}).get("temperature"))
                humidity = (_extract_float(text, "humidity", "humid") or
                            (params or {}).get("humidity"))
                rainfall = (_extract_float(text, "rainfall", "rain") or
                            (params or {}).get("rainfall"))

                # Only invoke if we have at least one numeric value
                if any(v is not None for v in [temp, humidity, rainfall]):
                    result = _build_suitability_response(crop, temp, humidity, rainfall)
                    if result:
                        return result

        # ── Rule 3: "Is it safe/good to plant" with temp/humidity in text ─────
        if _is_suitability_question(text):
            crop = _detect_crop(text)
            temp     = _extract_float(text, "temp", "temperature", r"\d+\s*°c")
            humidity = _extract_float(text, "humidity", "humid")
            rainfall = _extract_float(text, "rainfall", "rain")
            if crop and any(v is not None for v in [temp, humidity, rainfall]):
                result = _build_suitability_response(crop, temp, humidity, rainfall)
                if result:
                    return result

        return None   # nothing matched — fall through to chatbot

    def get_dosage(self, fertilizer: str, crop: str = "default") -> str:
        """Direct dosage lookup — usable from API endpoints."""
        return _build_dosage_response(fertilizer, crop)

    def evaluate_suitability(
        self,
        crop:     str,
        temp:     Optional[float] = None,
        humidity: Optional[float] = None,
        rainfall: Optional[float] = None,
    ) -> Optional[str]:
        """Direct suitability check — usable from API endpoints."""
        return _build_suitability_response(crop, temp, humidity, rainfall)

    def supported_crops(self) -> list:
        """Return list of crops the reasoning engine has thresholds for."""
        return sorted(CROP_THRESHOLDS.keys())

    def supported_fertilizers(self) -> list:
        """Return list of fertilizers the engine has dosage data for."""
        return sorted(DOSAGE_DB.keys())


# Singleton for import
reasoning_engine = ReasoningEngine()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = ReasoningEngine()

    print("=" * 70)
    print("REASONING ENGINE — SELF TEST")
    print("=" * 70)

    TESTS = [
        # (text, intent, expected_behaviour)
        ("How much urea should I use for rice per acre?",
         "fertilizer_advice",
         "Must return specific dosage (80-100 kg/acre), NOT generic NPK list"),

        ("How much urea for wheat?",
         "fertilizer_advice",
         "Must return wheat-specific urea dose"),

        ("How many kg of DAP for my cotton crop?",
         "fertilizer_advice",
         "Must return cotton DAP dosage"),

        ("30°C and 75% humidity — can I grow maize?",
         "weather_planting",
         "Must return maize suitability verdict for 30C/75%"),

        ("Temperature is 38°C and humidity 80%, planting rice?",
         "weather_planting",
         "Should flag 38°C as too hot for rice"),

        ("It is 12°C now. Should I plant rice?",
         "crop_recommendation",
         "Should flag 12°C as too cold for rice"),

        ("My soil has N=90, P=42, K=43 — what crop?",
         "crop_recommendation",
         "No numeric temp/humidity → should return None (fall through)"),

        ("What fertilizer for my wheat?",
         "fertilizer_advice",
         "No dosage question word → should return None (fall through)"),
    ]

    pass_count = 0
    for text, intent, expectation in TESTS:
        result = engine.intercept(text, intent)
        status = "✅ INTERCEPTED" if result else "⬇️  PASS-THROUGH"
        print(f"\n{status}")
        print(f"  Query:    {text}")
        print(f"  Intent:   {intent}")
        print(f"  Expected: {expectation}")
        if result:
            print(f"  Answer preview:\n    {result[:200].replace(chr(10), chr(10) + '    ')}...")
            pass_count += 1

    print("\n" + "=" * 70)
    print(f"Engine intercepted {pass_count}/{len(TESTS)} queries (expected: 6)")
    print(f"Supported crops:       {engine.supported_crops()}")
    print(f"Supported fertilizers: {engine.supported_fertilizers()}")
    print("=" * 70)
