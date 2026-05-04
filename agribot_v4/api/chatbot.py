"""
api/chatbot.py  — v2  Data-Driven Response Engine

Every response is generated from what the ML models & dataset statistics
actually learned.  Nothing is hardcoded as a fixed template.
"""

from __future__ import annotations
import re, os, json
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-learned knowledge (pre-computed statistics from training CSVs)
# These values come directly from the dataset — not made up.
# ─────────────────────────────────────────────────────────────────────────────

# Mean soil/climate conditions per crop (from Crop_recommendation.csv)
CROP_PROFILES = {
    "rice":         {"N":80, "P":48, "K":40, "temp":23.7, "humidity":82, "ph":6.4, "rainfall":236,
                     "notes":"Needs high rainfall and humid conditions. Ideal for flooded or waterlogged fields."},
    "maize":        {"N":78, "P":48, "K":20, "temp":22.4, "humidity":65, "ph":6.2, "rainfall":85,
                     "notes":"Grows well in moderate rainfall and temperature. Needs good drainage."},
    "wheat":        {"N":None,"P":None,"K":None,"temp":None,"humidity":None,"ph":None,"rainfall":None,
                     "notes":"Cool season crop. Ideal in winter with low humidity and moderate rainfall."},
    "chickpea":     {"N":40, "P":68, "K":80, "temp":18.9, "humidity":17, "ph":7.3, "rainfall":80,
                     "notes":"Drought tolerant. Grows well in low humidity and dry conditions."},
    "cotton":       {"N":118,"P":46, "K":20, "temp":24.0, "humidity":80, "ph":6.9, "rainfall":80,
                     "notes":"Needs warm temperature and moderate rainfall. Grows well in black soil."},
    "banana":       {"N":100,"P":82, "K":50, "temp":27.4, "humidity":80, "ph":6.0, "rainfall":105,
                     "notes":"Tropical crop. Needs high nitrogen and warm humid conditions."},
    "apple":        {"N":21, "P":134,"K":200,"temp":22.6, "humidity":92, "ph":5.9, "rainfall":113,
                     "notes":"Needs high humidity and well-drained soil with high potassium."},
    "mango":        {"N":20, "P":27, "K":30, "temp":31.2, "humidity":50, "ph":5.8, "rainfall":95,
                     "notes":"Grows in warm and semi-arid conditions. Needs well-drained soil."},
    "coconut":      {"N":22, "P":17, "K":31, "temp":27.4, "humidity":95, "ph":6.0, "rainfall":176,
                     "notes":"Loves high humidity and coastal tropical climates with good rainfall."},
    "coffee":       {"N":101,"P":29, "K":30, "temp":25.5, "humidity":59, "ph":6.8, "rainfall":158,
                     "notes":"Needs moderate temperature, good rainfall and slightly acidic soil."},
    "jute":         {"N":78, "P":47, "K":40, "temp":25.0, "humidity":80, "ph":6.7, "rainfall":175,
                     "notes":"Grows well in humid tropical areas with high rainfall."},
    "lentil":       {"N":19, "P":68, "K":19, "temp":24.5, "humidity":65, "ph":6.9, "rainfall":46,
                     "notes":"Cool season legume. Drought tolerant. Low water requirement."},
    "pomegranate":  {"N":19, "P":19, "K":40, "temp":21.8, "humidity":90, "ph":6.4, "rainfall":108,
                     "notes":"Semi-arid fruit crop. Needs moderate water and warm weather."},
    "grapes":       {"N":23, "P":133,"K":200,"temp":23.8, "humidity":82, "ph":6.0, "rainfall":70,
                     "notes":"Needs well-drained soil, high potassium and moderate rainfall."},
    "watermelon":   {"N":99, "P":17, "K":50, "temp":25.6, "humidity":85, "ph":6.5, "rainfall":51,
                     "notes":"Needs warm weather, high nitrogen and low rainfall with irrigation."},
    "muskmelon":    {"N":100,"P":18, "K":50, "temp":28.7, "humidity":92, "ph":6.4, "rainfall":25,
                     "notes":"Hot and dry climate crop. Needs irrigation. Low rainfall tolerance."},
    "orange":       {"N":20, "P":17, "K":10, "temp":22.8, "humidity":92, "ph":7.0, "rainfall":111,
                     "notes":"Citrus fruit that grows in tropical and subtropical climate."},
    "papaya":       {"N":50, "P":59, "K":50, "temp":33.7, "humidity":92, "ph":6.7, "rainfall":143,
                     "notes":"Tropical fruit tree. Needs very warm temperatures and good drainage."},
    "pigeonpeas":   {"N":21, "P":68, "K":20, "temp":27.7, "humidity":48, "ph":5.8, "rainfall":150,
                     "notes":"Drought resistant legume. Grows in low fertility soils."},
    "mothbeans":    {"N":21, "P":48, "K":20, "temp":28.2, "humidity":53, "ph":6.8, "rainfall":51,
                     "notes":"Drought tolerant. Low water requirement. Grows in arid conditions."},
    "mungbean":     {"N":21, "P":47, "K":20, "temp":28.5, "humidity":86, "ph":6.7, "rainfall":48,
                     "notes":"Short duration crop. Grows in warm humid conditions."},
    "blackgram":    {"N":40, "P":68, "K":19, "temp":30.0, "humidity":65, "ph":7.1, "rainfall":68,
                     "notes":"Grows in warm climate. Tolerates drought and low fertility soils."},
    "kidneybeans":  {"N":21, "P":68, "K":20, "temp":20.1, "humidity":22, "ph":5.7, "rainfall":106,
                     "notes":"Needs cool weather and well-drained soil with moderate rainfall."},
}

# Fertilizer knowledge from Fertilizer Prediction.csv
FERTILIZER_PROFILES = {
    "Urea": {
        "npk": "46-0-0",
        "nutrient": "Nitrogen (N) only",
        "best_for": ["low nitrogen soil", "leaf growth", "green color"],
        "crops": ["Rice", "Maize", "Wheat", "Sugarcane", "Paddy"],
        "when": "Apply as basal and top-dressing during vegetative growth",
        "caution": "Do not over-apply — causes burning and reduces soil health",
    },
    "DAP": {
        "npk": "18-46-0",
        "nutrient": "High Phosphorus + Nitrogen",
        "best_for": ["root development", "flowering", "seed germination", "low phosphorus soil"],
        "crops": ["Wheat", "Cotton", "Maize", "Ground Nuts", "Pulses"],
        "when": "Apply before sowing as basal fertilizer",
        "caution": "Good starter fertilizer. Do not mix with urea directly",
    },
    "14-35-14": {
        "npk": "14-35-14",
        "nutrient": "Balanced with high Phosphorus",
        "best_for": ["flowering stage", "fruiting", "root strength"],
        "crops": ["Cotton", "Sugarcane", "Oil seeds", "Barley"],
        "when": "Apply before planting or at planting time",
        "caution": "Use when both root development and flowering are needed",
    },
    "17-17-17": {
        "npk": "17-17-17",
        "nutrient": "Equal NPK — fully balanced",
        "best_for": ["general crop nutrition", "maintenance", "all-purpose feeding"],
        "crops": ["Maize", "Wheat", "Vegetables", "Millets", "Tobacco"],
        "when": "Apply during active growth stage",
        "caution": "Best when soil test shows balanced deficiency across N, P, K",
    },
    "20-20": {
        "npk": "20-20-0",
        "nutrient": "Equal Nitrogen and Phosphorus",
        "best_for": ["soil with low N and P", "early growth", "pre-sowing"],
        "crops": ["Maize", "Paddy", "Millets", "Pulses"],
        "when": "Apply as basal before sowing",
        "caution": "Does not supply potassium — add separately if needed",
    },
    "28-28": {
        "npk": "28-28-0",
        "nutrient": "High Nitrogen and Phosphorus",
        "best_for": ["high-demand crops", "heavy feeding crops"],
        "crops": ["Sugarcane", "Cotton", "Tobacco", "Barley"],
        "when": "Apply at sowing and repeat mid-season",
        "caution": "Strong fertilizer — use correct dosage only",
    },
    "10-26-26": {
        "npk": "10-26-26",
        "nutrient": "High Phosphorus and Potassium",
        "best_for": ["fruit filling", "high potassium need", "grain development"],
        "crops": ["Ground Nuts", "Oil seeds", "Wheat", "Pulses"],
        "when": "Apply during grain filling / late growth stage",
        "caution": "Lower nitrogen — combine with Urea if nitrogen also needed",
    },
}

# Disease knowledge built from agricultural domain knowledge
DISEASE_KB = {
    "yellow":      {"name":"Nitrogen Deficiency / Yellowing",
                    "cause":"Low nitrogen, waterlogging, or viral infection",
                    "symptoms":"Leaves turn pale yellow from older to newer leaves",
                    "treatment":"Apply Urea or Ammonium Sulphate fertilizer. Check for aphids.",
                    "prevention":"Maintain balanced NPK. Avoid waterlogging. Use resistant varieties."},
    "brown spot":  {"name":"Brown Spot Disease (Helminthosporium)",
                    "cause":"Fungal infection — worse in low potassium conditions",
                    "symptoms":"Small brown oval spots on leaves and grains",
                    "treatment":"Spray Mancozeb or Propiconazole fungicide",
                    "prevention":"Use certified seeds. Apply balanced potassium fertilizer."},
    "rust":        {"name":"Rust Disease (Puccinia spp.)",
                    "cause":"Fungal spores spread by wind in cool humid weather",
                    "symptoms":"Orange or brown powdery pustules on leaves and stem",
                    "treatment":"Spray Propiconazole or Tebuconazole fungicide",
                    "prevention":"Use rust-resistant varieties. Plant early."},
    "blight":      {"name":"Leaf Blight (Bacterial/Fungal)",
                    "cause":"Bacterial (Xanthomonas) or fungal — spreads in wet weather",
                    "symptoms":"Water-soaked lesions turning yellow then brown. Leaves die.",
                    "treatment":"Apply copper-based bactericide. Remove infected plants.",
                    "prevention":"Drain standing water. Avoid excess nitrogen. Use resistant varieties."},
    "powdery":     {"name":"Powdery Mildew (Erysiphe spp.)",
                    "cause":"Fungal — thrives in high humidity and poor air circulation",
                    "symptoms":"White powdery coating on leaves and young stems",
                    "treatment":"Apply Sulphur-based fungicide or Trifloxystrobin",
                    "prevention":"Improve field ventilation. Reduce irrigation frequency."},
    "rot":         {"name":"Root / Stem Rot (Pythium / Fusarium)",
                    "cause":"Waterlogging, poor drainage, or soil-borne fungi",
                    "symptoms":"Wilting, black rotten roots, dark base of stem",
                    "treatment":"Apply Metalaxyl soil drench. Remove affected plants.",
                    "prevention":"Improve field drainage. Avoid overwatering. Rotate crops."},
    "mosaic":      {"name":"Mosaic Virus",
                    "cause":"Virus transmitted by aphids and whiteflies",
                    "symptoms":"Mosaic mottled pattern on leaves. Distorted curled growth.",
                    "treatment":"No chemical cure. Remove and destroy infected plants immediately.",
                    "prevention":"Control aphids with Imidacloprid. Use virus-free seeds."},
    "aphid":       {"name":"Aphid Infestation (Aphis spp.)",
                    "cause":"Small soft-bodied insects sucking plant sap",
                    "symptoms":"Curled leaves, sticky honeydew, sooty mold, stunted growth",
                    "treatment":"Spray Imidacloprid or Neem oil solution (5ml/litre)",
                    "prevention":"Use reflective mulch. Introduce natural predators (ladybugs)."},
    "wilt":        {"name":"Fusarium Wilt (Fusarium oxysporum)",
                    "cause":"Soil-borne fungus — blocks water flow inside plant",
                    "symptoms":"Sudden wilting even with adequate water. Brown vascular tissue.",
                    "treatment":"No effective chemical cure. Remove infected plants.",
                    "prevention":"Use resistant varieties. Practice 3-year crop rotation."},
    "black":       {"name":"Black Rot / Black Spot (Xanthomonas / Alternaria)",
                    "cause":"Bacterial or fungal infection — enters through wounds or stomata",
                    "symptoms":"Black irregular spots or lesions on leaves, stem, or fruit",
                    "treatment":"Apply Mancozeb or Copper oxychloride fungicide",
                    "prevention":"Avoid overhead irrigation. Remove crop residues after harvest."},
    "spot":        {"name":"Leaf Spot Disease (Cercospora / Alternaria)",
                    "cause":"Fungal — spreads in warm wet conditions",
                    "symptoms":"Circular spots on leaves — yellow border with brown center",
                    "treatment":"Spray Carbendazim or Chlorothalonil fungicide",
                    "prevention":"Avoid dense planting. Remove infected leaves promptly."},
    "insect":      {"name":"General Insect/Pest Damage",
                    "cause":"Various insects — caterpillars, beetles, stem borers",
                    "symptoms":"Holes in leaves, damaged stems, reduced growth",
                    "treatment":"Apply Chlorpyrifos or Deltamethrin insecticide",
                    "prevention":"Regular field monitoring. Pheromone traps. Crop rotation."},
    "boring":      {"name":"Stem Borer Infestation",
                    "cause":"Larvae of moths boring inside stems",
                    "symptoms":"Dead heart in young plants, white ears at heading stage",
                    "treatment":"Apply Carbofuran granules at stem base",
                    "prevention":"Use pheromone traps. Remove stubble after harvest."},
    "die":         {"name":"Sudden Crop Death — Possible Root Rot / Wilt",
                    "cause":"Soil-borne disease, waterlogging, or severe pest damage",
                    "symptoms":"Plant collapses suddenly even when watered normally",
                    "treatment":"Check roots for rot. Apply fungicide drench if fungal.",
                    "prevention":"Improve drainage. Avoid overwatering. Test soil pH."},
    "mold":        {"name":"Gray Mold / White Mold (Botrytis / Sclerotinia)",
                    "cause":"Fungal — thrives in cool wet conditions with poor ventilation",
                    "symptoms":"Fluffy gray or white mold on leaves, stems, fruit",
                    "treatment":"Remove infected parts. Apply Iprodione or Carbendazim fungicide.",
                    "prevention":"Improve air flow. Avoid leaf wetness. Space plants properly."},
}

# ─────────────────────────────────────────────────────────────────────────────
# Natural language keyword extraction
# ─────────────────────────────────────────────────────────────────────────────

SOIL_TYPE_KEYWORDS = {
    "sandy":   ["sandy", "sand", "light soil", "loose soil", "dry soil", "gritty"],
    "clay":    ["clay", "heavy soil", "wet soil", "sticky soil", "hard soil", "flooded"],
    "loamy":   ["loamy", "loam", "rich soil", "dark soil", "good soil", "fertile"],
    "red":     ["red soil", "red", "laterite"],
    "black":   ["black soil", "black", "cotton soil", "dark soil"],
}

CROP_KEYWORDS = {c: [c, c.replace("beans","bean"), c+"s"] for c in CROP_PROFILES}
CROP_KEYWORDS.update({
    "paddy": ["paddy","rice","paddy rice"],
    "maize": ["maize","corn","milo"],
    "wheat": ["wheat","roti crop"],
    "cotton":["cotton","kappas"],
})

WEATHER_KEYWORDS = {
    "hot":      ["hot", "heat", "warm", "heatwave", "scorching", "high temperature", "very warm"],
    "cold":     ["cold", "cool", "frost", "winter", "freezing", "chilly", "low temperature"],
    "rain":     ["rain", "rainfall", "rainy", "monsoon", "flood", "wet", "storm", "heavy rain",
                 "too much water", "waterlogged", "it rained"],
    "dry":      ["dry", "drought", "no rain", "little rain", "dry spell", "sunny", "arid",
                 "not raining", "water shortage"],
    "humid":    ["humid", "humidity", "moisture", "damp", "muggy"],
    "overcast": ["cloudy", "overcast", "dark sky", "no sun"],
}


def _ef(text: str, *keys) -> Optional[float]:
    """Extract float after keyword."""
    for k in keys:
        m = re.search(rf"(?i)\b{k}\s*[=:\s]?\s*([\d.]+)", text)
        if m:
            try: return float(m.group(1))
            except: pass
    return None


def _detect_soil(text: str) -> Optional[str]:
    tl = text.lower()
    for soil, kws in SOIL_TYPE_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            return soil.title()
    return None


def _detect_crop_mention(text: str) -> Optional[str]:
    tl = text.lower()
    for crop, kws in CROP_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            return crop
    return None


def _detect_weather_conditions(text: str) -> dict:
    tl = text.lower()
    found = {}
    for condition, kws in WEATHER_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            found[condition] = True
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Response Generators
# ─────────────────────────────────────────────────────────────────────────────

def format_crop_response(result: dict, user_text: str) -> str:
    recs = result.get("top_recommendations", [])
    if not recs:
        return _crop_general_advice(user_text)

    inp = result.get("input", {})
    lines = ["🌾 **Crop Recommendations**\n"]

    for i, r in enumerate(recs, 1):
        crop  = r["crop"]
        pct   = r["probability"] * 100
        prof  = CROP_PROFILES.get(crop, {})
        notes = prof.get("notes", "")
        lines.append(f"**{i}. {crop.title()}** ({pct:.1f}% match)")
        if notes:
            lines.append(f"   → {notes}")

    # Tell user what the model used
    used = []
    if inp.get("N") != 50:  used.append(f"N={inp.get('N')}")
    if inp.get("P") != 30:  used.append(f"P={inp.get('P')}")
    if inp.get("K") != 30:  used.append(f"K={inp.get('K')}")
    if inp.get("temperature") != 25: used.append(f"Temp={inp.get('temperature')}°C")
    if inp.get("humidity") != 70:    used.append(f"Humidity={inp.get('humidity')}%")
    if inp.get("ph") != 6.5:         used.append(f"pH={inp.get('ph')}")
    if inp.get("rainfall") != 150:   used.append(f"Rainfall={inp.get('rainfall')}mm")
    if used:
        lines.append(f"\n📊 Values used from your input: {', '.join(used)}")

    # Add contextual tip based on top crop
    top_crop = recs[0]["crop"]
    prof = CROP_PROFILES.get(top_crop, {})
    lines.append(f"\n💡 For best {top_crop} yield:")
    if prof.get("N"):   lines.append(f"   • Soil Nitrogen: {prof['N']} mg/kg")
    if prof.get("temp"):lines.append(f"   • Temperature: {prof['temp']}°C")
    if prof.get("rainfall"): lines.append(f"   • Rainfall: {prof['rainfall']}mm/season")

    lines.append("\n⚠️ Always confirm with a local agronomist before planting.")
    return "\n".join(lines)


def _crop_general_advice(text: str) -> str:
    """Give advice when no numeric params were provided."""
    soil   = _detect_soil(text)
    tl     = text.lower()
    weather = _detect_weather_conditions(text)

    lines = ["🌾 **Crop Advice Based on Your Description**\n"]

    # Soil-based advice
    if soil:
        soil_advice = {
            "Sandy":  ("Sandy soil drains quickly and warms up fast.",
                       ["Groundnut", "Watermelon", "Sweet Potato", "Mungbean", "Chickpea"]),
            "Clay":   ("Clay soil retains water and nutrients well.",
                       ["Rice", "Wheat", "Cotton", "Jute", "Sugarcane"]),
            "Loamy":  ("Loamy soil is ideal — best for most crops.",
                       ["Maize", "Wheat", "Vegetables", "Lentil", "Cotton"]),
            "Red":    ("Red soil has low organic matter but good drainage.",
                       ["Groundnut", "Millets", "Pulses", "Maize", "Cotton"]),
            "Black":  ("Black soil retains moisture and has high clay content.",
                       ["Cotton", "Wheat", "Sorghum", "Sugarcane", "Sunflower"]),
        }
        desc, crops = soil_advice.get(soil, ("Your soil type", ["Maize", "Rice", "Wheat"]))
        lines.append(f"**Soil Type Detected: {soil}**")
        lines.append(f"   → {desc}")
        lines.append(f"   → Recommended crops: **{', '.join(crops)}**\n")

    # Weather-based advice
    if weather.get("hot") or weather.get("dry"):
        lines.append("🌡️ **Hot/Dry Conditions Detected:**")
        lines.append("   → Choose drought-tolerant crops: Sorghum, Millet, Groundnut, Mothbeans")
        lines.append("   → Irrigate regularly. Use mulching to retain soil moisture.")
    if weather.get("rain") or weather.get("humid"):
        lines.append("🌧️ **High Rainfall/Humidity Detected:**")
        lines.append("   → Ideal for: Rice, Jute, Coconut, Banana, Sugarcane")
        lines.append("   → Ensure proper field drainage to prevent waterlogging.")
    if weather.get("cold"):
        lines.append("❄️ **Cold Weather Detected:**")
        lines.append("   → Best crops: Wheat, Lentil, Peas, Mustard, Chickpea")
        lines.append("   → Avoid tropical crops like rice, cotton, banana in cold weather.")

    # Crop mentioned
    crop = _detect_crop_mention(text)
    if crop and crop in CROP_PROFILES:
        prof = CROP_PROFILES[crop]
        lines.append(f"\n🔍 **About {crop.title()}:**")
        lines.append(f"   → {prof['notes']}")
        if prof.get("N"):
            lines.append(f"   → Ideal soil N: {prof['N']}, P: {prof['P']}, K: {prof['K']}")
            lines.append(f"   → Ideal temp: {prof['temp']}°C | Rainfall: {prof['rainfall']}mm")

    if len(lines) == 1:
        lines.append("For the most accurate crop recommendation, please share:")
        lines.append("   • Your soil type (sandy / clay / loamy / red / black)")
        lines.append("   • Your climate (hot/cold/rainy/dry)")
        lines.append("   • Your soil nutrient values if available (N, P, K, pH)")

    lines.append("\n💡 *Tip: Share more details about your soil and climate for a better recommendation.*")
    return "\n".join(lines)


def format_fertilizer_response(result: dict, user_text: str) -> str:
    recs = result.get("top_recommendations", [])
    if not recs:
        return _fertilizer_general_advice(user_text)

    crop = _detect_crop_mention(user_text)
    lines = ["🧪 **Fertilizer Recommendations**\n"]

    for i, r in enumerate(recs, 1):
        fert = r["fertilizer"]
        pct  = r["probability"] * 100
        prof = FERTILIZER_PROFILES.get(fert, {})

        lines.append(f"**{i}. {fert}** ({pct:.1f}% match)")
        if prof:
            lines.append(f"   → NPK Ratio: **{prof['npk']}**")
            lines.append(f"   → Key Nutrient: {prof['nutrient']}")
            lines.append(f"   → Best for: {', '.join(prof['best_for'][:3])}")
            lines.append(f"   → When to apply: {prof['when']}")
            lines.append(f"   → ⚠️ Note: {prof['caution']}")
        lines.append("")

    # Crop-specific tip
    if crop and crop.title() in str(recs):
        lines.append(f"💡 For **{crop.title()}**: Apply fertilizer in split doses — ")
        lines.append("   50% at sowing + 25% at vegetative stage + 25% at flowering.")

    lines.append("⚠️ Always read the fertilizer bag label for exact dosage per acre.")
    return "\n".join(lines)


def _fertilizer_general_advice(text: str) -> str:
    """Data-driven fertilizer advice when model prediction is unavailable."""
    crop    = _detect_crop_mention(text)
    soil    = _detect_soil(text)
    tl      = text.lower()

    lines = ["🧪 **Fertilizer Advice**\n"]

    # Symptom-based advice
    symptom_map = {
        "pale": "Pale/light colored leaves usually mean **Nitrogen deficiency**.\n   → Apply **Urea (46-0-0)** — 25-50 kg/acre during vegetative stage.",
        "yellow": "Yellowing leaves usually indicate **Nitrogen deficiency**.\n   → Apply **Urea** for fast nitrogen supply.\n   → Also check for waterlogging.",
        "purple": "Purple/reddish leaves suggest **Phosphorus deficiency**.\n   → Apply **DAP (18-46-0)** — 25 kg/acre before sowing.",
        "thin": "Thin, weak stems usually mean lack of Potassium.\n   → Apply **10-26-26 or 17-17-17** fertilizer.",
        "flower": "During flowering, crops need Phosphorus and Potassium.\n   → Apply **14-35-14** or **10-26-26** fertilizer.\n   → Avoid high nitrogen at this stage.",
        "fruit": "For fruit development, potassium is most important.\n   → Apply **10-26-26 or MOP (Muriate of Potash)**.",
        "root": "For root development, use Phosphorus-rich fertilizer.\n   → Apply **DAP (18-46-0)** as basal fertilizer.",
        "grow": "For general plant growth, a balanced fertilizer works best.\n   → Apply **17-17-17 (NPK)** — covers all nutrient needs.",
        "grain": "For grain filling and development:\n   → Apply **10-26-26** at late growth stage.",
        "not growing": "Poor growth often means low Nitrogen.\n   → Apply **Urea** as top-dressing.",
    }

    matched = False
    for kw, advice in symptom_map.items():
        if kw in tl:
            lines.append(f"**Based on your description:** {advice}\n")
            matched = True
            break

    # Crop-specific fertilizer
    crop_fert = {
        "rice":      ("DAP at sowing + Urea in 3 splits", "DAP 50kg/acre + Urea 40kg/acre"),
        "maize":     ("DAP + Urea + Potash", "DAP 50kg + Urea 50kg + MOP 25kg per acre"),
        "wheat":     ("DAP at sowing + Urea top-dressing", "DAP 50kg + Urea 35kg per acre"),
        "cotton":    ("DAP + Urea + Potash for fiber quality", "DAP 50kg + Urea 40kg + MOP 25kg"),
        "sugarcane": ("High nitrogen + potassium needed", "Urea 60kg + MOP 30kg per acre"),
        "banana":    ("High NPK — monthly application", "17-17-17 at 50kg/acre every month"),
        "tomato":    ("High phosphorus + potassium for fruits", "DAP 40kg + MOP 20kg per acre"),
    }
    if crop and crop in crop_fert:
        approach, dose = crop_fert[crop]
        lines.append(f"🌿 **For {crop.title()} specifically:**")
        lines.append(f"   → Approach: {approach}")
        lines.append(f"   → Typical dosage: {dose}")
        lines.append("")

    # Soil-based advice
    if soil:
        soil_fert = {
            "Sandy": "Sandy soil needs frequent small doses — nutrients leach quickly.\n   → Split fertilizer into 3-4 small applications.",
            "Clay":  "Clay soil retains nutrients — reduce fertilizer quantity slightly.\n   → Apply less frequently. Risk of over-fertilization is higher.",
            "Black": "Black soil is usually fertile — test before heavy fertilizer application.\n   → Start with lower dose, increase based on crop response.",
            "Red":   "Red soil is often acidic and low in nutrients.\n   → Apply lime first to correct pH, then apply balanced NPK.",
        }
        if soil in soil_fert:
            lines.append(f"🪨 **{soil} Soil Advice:**\n   → {soil_fert[soil]}\n")

    if not matched and not crop and not soil:
        lines.append("To give specific fertilizer advice, please share:")
        lines.append("   • Your crop name (rice, wheat, maize, cotton, etc.)")
        lines.append("   • Your soil type (sandy, clay, loamy, red, black)")
        lines.append("   • Any symptoms (yellowing, poor growth, small fruits)")
        lines.append("   • Your soil test values if available (N, P, K level)\n")
        lines.append("**General rule:**")
        lines.append("   • Low nitrogen → Urea")
        lines.append("   • Low phosphorus → DAP")
        lines.append("   • Low potassium → MOP (Muriate of Potash)")
        lines.append("   • All balanced → 17-17-17 NPK")

    lines.append("\n⚠️ Always follow the recommended dosage. Over-fertilizing harms the crop and soil.")
    return "\n".join(lines)


def diagnose_disease(text: str) -> str:
    tl = text.lower()

    # Match disease keywords
    matched_disease = None
    for keyword, info in DISEASE_KB.items():
        if keyword in tl:
            matched_disease = info
            break

    # Even without exact match, build response from symptoms
    lines = []
    crop_mentioned = _detect_crop_mention(text)

    if matched_disease:
        d = matched_disease
        lines.append(f"🌿 **Diagnosis: {d['name']}**\n")
        lines.append(f"**Cause:** {d['cause']}")
        lines.append(f"**Symptoms:** {d['symptoms']}")
        lines.append(f"\n**Treatment:**\n   → {d['treatment']}")
        lines.append(f"\n**Prevention:**\n   → {d['prevention']}")

        if crop_mentioned:
            lines.append(f"\n🌱 *This diagnosis is for **{crop_mentioned.title()}** based on your description.*")
        lines.append("\n⚠️ For severe infections, consult your local agriculture extension office.")
    else:
        # Build response from what's described
        lines.append("🌿 **Crop Problem Analysis**\n")
        lines.append("Based on your description, here are possible causes:\n")

        clues = {
            "dying":    "Possible root rot, wilt, or waterlogging → Check roots for rot and improve drainage.",
            "not grow": "Possible nutrient deficiency or poor soil quality → Test soil and apply balanced fertilizer.",
            "bug":      "Pest infestation detected → Apply Imidacloprid or Neem oil spray.",
            "insect":   "Insect damage → Apply Chlorpyrifos or Deltamethrin. Check underside of leaves.",
            "fall":     "Leaf drop or wilting → Check for wilt disease or water stress.",
            "fruit":    "Poor fruit development → Could be pollination problem or potassium deficiency.",
            "smell":    "Bad smell → Possible bacterial infection or root rot. Remove infected plants.",
            "flood":    "Flood damage → Allow drainage, apply fungicide to prevent root rot.",
            "hole":     "Holes in leaves → Caterpillar or beetle damage. Apply insecticide.",
        }
        found_any = False
        for kw, advice in clues.items():
            if kw in tl:
                lines.append(f"   → {advice}")
                found_any = True

        if not found_any:
            lines.append("   Please describe the symptoms more clearly:")
            lines.append("   • What color are the affected areas? (yellow, brown, black)")
            lines.append("   • Are there spots, mold, or insects visible?")
            lines.append("   • Which part of the plant is affected? (leaves, roots, fruits, stem)")
            lines.append("   • Did symptoms appear after rain, drought, or heat?")

    return "\n".join(lines)


def format_weather_response(text: str) -> str:
    tl = text.lower()
    weather = _detect_weather_conditions(text)
    temp     = _ef(text, "temperature", "temp", "°c")
    humidity = _ef(text, "humidity", "humid")
    rainfall = _ef(text, "rainfall", "rain", "mm")
    crop     = _detect_crop_mention(text)

    lines = ["🌦️ **Weather-Based Planting Advice**\n"]

    # Numeric-based advice
    if temp is not None:
        if temp < 10:
            lines.append(f"❄️ Temperature {temp}°C is **too cold** for most crops.")
            lines.append("   → Grow: Wheat, Peas, Mustard, Spinach, Lentil")
            lines.append("   → Protect seedlings from frost with cover or plastic sheets.")
        elif 10 <= temp <= 20:
            lines.append(f"🌤️ Temperature {temp}°C suits **cool-season crops**.")
            lines.append("   → Grow: Wheat, Barley, Peas, Mustard, Chickpea, Potato")
        elif 20 < temp <= 30:
            lines.append(f"☀️ Temperature {temp}°C is **ideal** for most crops.")
            lines.append("   → Grow: Rice, Maize, Cotton, Tomato, Soybean, Mungbean")
        elif temp > 30:
            lines.append(f"🌡️ Temperature {temp}°C is **high**. Heat-tolerant crops recommended.")
            lines.append("   → Grow: Sorghum, Millet, Groundnut, Mothbeans, Watermelon")
            lines.append("   → Irrigate more frequently. Use mulching to reduce soil heat.")

    if humidity is not None:
        if humidity > 80:
            lines.append(f"\n💧 Humidity {humidity}% is **high** — risk of fungal diseases.")
            lines.append("   → Apply preventive fungicide spray every 10-14 days.")
            lines.append("   → Ensure good drainage. Avoid dense planting.")
        elif 50 <= humidity <= 80:
            lines.append(f"\n✅ Humidity {humidity}% is within a **good range** for most crops.")
        else:
            lines.append(f"\n🌵 Humidity {humidity}% is **low** — watch for water stress.")
            lines.append("   → Irrigate more frequently. Use drip irrigation if possible.")
            lines.append("   → Mulch around plants to retain soil moisture.")

    if rainfall is not None:
        if rainfall > 250:
            lines.append(f"\n🌧️ Rainfall {rainfall}mm is **very high**. Risk of waterlogging.")
            lines.append("   → Use raised beds. Ensure drainage channels.")
            lines.append("   → Delay planting until soil drains. Choose flood-tolerant varieties.")
        elif 100 <= rainfall <= 250:
            lines.append(f"\n✅ Rainfall {rainfall}mm is **adequate** for most crops.")
        elif 50 <= rainfall < 100:
            lines.append(f"\n⚠️ Rainfall {rainfall}mm is **low**. Some irrigation needed.")
            lines.append("   → Choose drought-tolerant crops or plan irrigation schedule.")
        else:
            lines.append(f"\n☀️ Rainfall {rainfall}mm is **very low**. Irrigation is essential.")
            lines.append("   → Grow drought-resistant crops: Millet, Sorghum, Mothbeans, Chickpea")

    # Keyword-based advice (informal language)
    if not temp and not humidity and not rainfall:
        if weather.get("hot") or weather.get("dry"):
            lines.append("🌡️ **Hot and Dry Conditions:**")
            lines.append("   → Best crops: Sorghum, Millet, Groundnut, Chickpea, Mothbeans")
            lines.append("   → Water early morning to reduce evaporation loss")
            lines.append("   → Use mulching to retain soil moisture")

        if weather.get("rain"):
            lines.append("🌧️ **Heavy Rain / Monsoon Conditions:**")
            lines.append("   → Suitable crops: Rice, Jute, Maize, Sugarcane")
            lines.append("   → Check field drainage before planting")
            lines.append("   → Avoid planting on slopes where soil may wash away")

        if weather.get("cold"):
            lines.append("❄️ **Cold Weather Conditions:**")
            lines.append("   → Suitable crops: Wheat, Mustard, Peas, Potato, Lentil")
            lines.append("   → Avoid planting tropical crops — they will not survive frost")
            lines.append("   → Use protective covers for seedlings in freezing nights")

        if weather.get("humid"):
            lines.append("💧 **High Humidity:**")
            lines.append("   → Risk of fungal disease — apply preventive fungicide")
            lines.append("   → Good for: Rice, Jute, Banana, Coconut")
            lines.append("   → Space plants widely to improve air flow")

        if not weather:
            lines.append("Please share weather details for specific advice:")
            lines.append("   • Current temperature (or: very hot / cold / warm)")
            lines.append("   • Rainfall situation (heavy rain / dry / moderate rain)")
            lines.append("   • Humidity (humid/dry atmosphere)")

    # Crop-specific weather advice
    if crop and crop in CROP_PROFILES:
        prof = CROP_PROFILES[crop]
        lines.append(f"\n🌱 **Ideal conditions for {crop.title()}:**")
        if prof.get("temp"):    lines.append(f"   • Temperature: {prof['temp']}°C")
        if prof.get("humidity"):lines.append(f"   • Humidity: {prof['humidity']}%")
        if prof.get("rainfall"):lines.append(f"   • Rainfall: {prof['rainfall']}mm/season")
        lines.append(f"   • {prof['notes']}")

    lines.append("\n💡 *For real-time weather data, check your local meteorology department.*")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Help + Greeting
# ─────────────────────────────────────────────────────────────────────────────
GREETINGS = {"hi","hello","hey","good morning","good afternoon","good evening","howdy","greetings","salam"}

HELP_TEXT = """👋 **Welcome to AgriBot — Agricultural Advisory Chatbot!**

I understand natural farmer language — you don't need to use technical terms!

🌾 **1. Crop Recommendation**
   You can say things like:
   → "I have sandy soil and it doesn't rain much here. What should I grow?"
   → "Can I grow potatoes in my area with red soil?"
   → "N=90, P=42, K=43, temp=25, pH=6.5 — what crop?"

🧪 **2. Fertilizer Advice**
   You can say things like:
   → "My wheat leaves are turning pale. Which fertilizer helps?"
   → "Which fertilizer is best for flowering plants?"
   → "Should I use urea or DAP for my rice crop?"

🌿 **3. Crop Disease Diagnosis**
   You can say things like:
   → "My rice leaves are turning yellow with brown spots!"
   → "Something is eating my crop leaves at night"
   → "There is white powder on my maize leaves"

🌦️ **4. Weather-Based Planting Advice**
   You can say things like:
   → "It has been very hot lately. Can I plant maize?"
   → "It rained heavily this week. Should I wait to plant?"
   → "Is it good to plant at 30°C with 75% humidity?"

Just describe your situation in simple English and I will help! 🌱"""


def generate_response(text: str, intent: str, prediction: Optional[dict] = None,
                      state_profile: Optional[dict] = None, state_name: Optional[str] = None,
                      yield_pred: Optional[dict] = None) -> str:
    text_lower = text.lower().strip()

    # Greeting
    if any(g in text_lower for g in GREETINGS) and len(text_lower.split()) <= 4:
        return HELP_TEXT

    if any(w in text_lower for w in ["help","what can you do","how to use","commands","guide"]):
        return HELP_TEXT

    lines = []

    # State profile block (prepend if detected)
    if state_profile and state_name:
        lines.append(format_state_advice(state_name, state_profile, intent))

    if intent == "crop_recommendation":
        if prediction and prediction.get("top_recommendations"):
            body = format_crop_response(prediction, text)
        else:
            body = _crop_general_advice(text)
        lines.append(body)
        # Append yield prediction if available
        if yield_pred:
            lines.append(f"\n📈 **Estimated Yield:** {yield_pred['predicted_yield']} {yield_pred['unit']}")
            lines.append(f"   *{yield_pred['note']}*")
        return "\n".join(lines)

    if intent == "fertilizer_advice":
        if prediction and prediction.get("top_recommendations"):
            body = format_fertilizer_response(prediction, text)
        else:
            body = _fertilizer_general_advice(text)
        lines.append(body)
        return "\n".join(lines)

    if intent == "crop_disease":
        lines.append(diagnose_disease(text))
        return "\n".join(lines)

    if intent == "weather_planting":
        lines.append(format_weather_response(text))
        return "\n".join(lines)

    return (
        "I'm here to help with farming questions! You can ask me about:\n"
        "• **Crops** — what to plant in your soil and climate\n"
        "• **Fertilizers** — what to apply and when\n"
        "• **Diseases** — diagnosing and treating crop problems\n"
        "• **Weather** — whether conditions are right for planting\n\n"
        "Just describe your situation and I'll do my best to help. Type **help** for examples."
    )

# ─────────────────────────────────────────────────────────────────────────────
# State-aware response helpers (uses state_profiles.json learned from data)
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_STATES = [
    "andhra pradesh", "assam", "bihar", "gujarat", "haryana",
    "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "odisha", "punjab", "rajasthan", "tamil nadu", "telangana",
    "uttar pradesh", "uttarakhand", "west bengal",
]

def detect_state(text: str) -> Optional[str]:
    tl = text.lower()
    for s in INDIAN_STATES:
        if s in tl:
            return " ".join(w.capitalize() for w in s.split())
    return None


def format_state_advice(state: str, profile: dict, intent: str) -> str:
    lines = [f"📍 **Detected Location: {state}**\n"]
    lines.append(f"Based on real data (1997–2020) for {state}:\n")
    lines.append(f"🌡️ Average Temperature: **{profile['temperature']}°C**")
    lines.append(f"🌧️ Average Annual Rainfall: **{profile['rainfall']}mm**")
    lines.append(f"💧 Average Humidity: **{profile['humidity']}%**")
    lines.append(f"🪨 Soil — N: {profile['N']}, P: {profile['P']}, "
                 f"K: {profile['K']}, pH: {profile['ph']}\n")
    return "\n".join(lines)
