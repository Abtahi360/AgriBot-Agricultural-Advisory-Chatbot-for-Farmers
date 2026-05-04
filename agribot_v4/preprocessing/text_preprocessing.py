"""
preprocessing/text_preprocessing.py

Complete text preprocessing pipeline for agricultural queries:
- Lowercasing
- Punctuation removal
- Whitespace normalization
- Spell variation handling
- Agricultural keyword preservation
"""

import re
import string
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


# Agricultural keywords that should be preserved/protected during preprocessing
AGRICULTURAL_KEYWORDS = {
    'rice', 'wheat', 'maize', 'corn', 'paddy', 'barley', 'millets',
    'cotton', 'sugarcane', 'potato', 'tomato', 'carrot', 'onion',
    'nitrogen', 'phosphorus', 'phosphorous', 'potassium', 'npk',
    'urea', 'dap', 'fertilizer', 'compost', 'manure',
    'soil', 'ph', 'humidity', 'temperature', 'rainfall', 'moisture',
    'pest', 'disease', 'fungal', 'bacterial', 'viral', 'blight',
    'yield', 'crop', 'farming', 'field', 'farm', 'harvest',
    'drought', 'flood', 'monsoon', 'season', 'weather',
    'sandy', 'loamy', 'clay', 'black', 'red', 'acidic', 'alkaline'
}

# Common spelling variations to normalize
SPELLING_VARIATIONS = {
    'phosphorous': 'phosphorus',
    'fertlizer': 'fertilizer',
    'yeild': 'yield',
    'desease': 'disease',
    'fert': 'fertilizer',
    'npk': 'npk',
    'n.p.k': 'npk',
}


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str, preserve_dots: bool = False) -> str:
    """
    Remove punctuation while optionally preserving dots for decimal numbers.
    
    Args:
        text: Input text
        preserve_dots: If True, preserve dots (for pH values like 6.5)
    
    Returns:
        Text with punctuation removed
    """
    if preserve_dots:
        # Keep dots and remove other punctuation
        return re.sub(r'[^\w\s.]', ' ', text)
    else:
        # Remove all punctuation
        return text.translate(str.maketrans('', '', string.punctuation))


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace:
    - Remove leading/trailing whitespace
    - Collapse multiple spaces into single space
    - Remove tabs and newlines
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    return text


def normalize_spelling(text: str) -> str:
    """
    Normalize common spelling variations.
    
    Example:
        'phosphorous deficiency' -> 'phosphorus deficiency'
    """
    words = text.split()
    normalized = []
    
    for word in words:
        # Check if exact word matches a variation
        if word in SPELLING_VARIATIONS:
            normalized.append(SPELLING_VARIATIONS[word])
        else:
            # Check for partial matches (e.g., word contains variation)
            found = False
            for variant, correct in SPELLING_VARIATIONS.items():
                if variant in word:
                    word = word.replace(variant, correct)
                    found = True
                    break
            normalized.append(word)
    
    return ' '.join(normalized)


def remove_extra_articles(text: str, aggressive: bool = False) -> str:
    """
    Remove unnecessary articles and stop words (optional).
    
    Args:
        text: Input text
        aggressive: If True, remove more stop words (may lose meaning)
    
    Returns:
        Text with some articles removed
    """
    if not aggressive:
        # Minimal removal: only remove duplicated articles
        text = re.sub(r'\ba\s+a\b', 'a', text)
        text = re.sub(r'\bthe\s+the\b', 'the', text)
        return text
    
    # Aggressive removal (be careful!)
    stop_words = {'a', 'an', 'the', 'is', 'are', 'am', 'i', 'you', 'he', 'she'}
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)


def normalize_numbers(text: str) -> str:
    """
    Standardize number formats:
    - Ensure consistent spacing around numbers
    - Handle ranges (e.g., "20-30" -> "20 to 30")
    """
    # Replace hyphens in ranges with 'to' (e.g., "20-30°C" -> "20 to 30°C")
    text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
    
    # Ensure space around degree symbol
    text = re.sub(r'(\d+)°', r'\1 degrees ', text)
    
    return text


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    normalize_ws: bool = True,
    normalize_spell: bool = True,
    normalize_nums: bool = True,
    preserve_decimals: bool = True,
) -> str:
    """
    Main preprocessing function: apply all transformations in sequence.
    
    Args:
        text: Raw input text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        normalize_ws: Normalize whitespace
        normalize_spell: Normalize spelling variations
        normalize_nums: Normalize number formats
        preserve_decimals: Keep decimal points (for pH, etc.)
    
    Returns:
        Fully preprocessed text
    
    Example:
        >>> text = "What fertilizer for my Phosphorous deficiency???"
        >>> preprocess_text(text)
        'what fertilizer for my phosphorus deficiency'
    """
    original = text
    
    if lowercase:
        text = lowercase_text(text)
    
    if normalize_spell:
        text = normalize_spelling(text)
    
    if remove_punct:
        text = remove_punctuation(text, preserve_dots=preserve_decimals)
    
    if normalize_nums:
        text = normalize_numbers(text)
    
    if normalize_ws:
        text = normalize_whitespace(text)
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words while preserving agricultural keywords.
    
    Args:
        text: Preprocessed text
    
    Returns:
        List of tokens
    """
    return text.split()


def extract_numbers(text: str) -> Dict[str, List[float]]:
    """
    Extract numerical values from text.
    Useful for feature extraction from queries like "N=90, P=42".
    
    Args:
        text: Input text
    
    Returns:
        Dict with 'values' list and 'units' list
    """
    # Find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    values = [float(n) for n in numbers]
    
    # Extract units if present
    units = []
    for match in re.finditer(r'(\d+\.?\d*)\s*([a-z%°]+)?', text, re.IGNORECASE):
        if match.group(2):
            units.append(match.group(2))
    
    return {
        'values': values,
        'units': units,
        'count': len(values)
    }


def detect_keywords(text: str) -> Dict[str, List[str]]:
    """
    Detect and categorize agricultural keywords in text.
    
    Args:
        text: Preprocessed text
    
    Returns:
        Dict with keyword categories
    """
    tokens = tokenize_text(text)
    
    crops = []
    nutrients = []
    fertilizers = []
    soil_types = []
    diseases = []
    
    crop_list = {'rice', 'wheat', 'maize', 'corn', 'paddy', 'barley', 
                 'millets', 'cotton', 'sugarcane', 'potato', 'tomato'}
    nutrient_list = {'nitrogen', 'phosphorus', 'phosphorous', 'potassium', 'npk'}
    fert_list = {'urea', 'dap', 'fertilizer', 'compost', 'manure'}
    soil_list = {'sandy', 'loamy', 'clay', 'black', 'red', 'acidic', 'alkaline'}
    disease_list = {'pest', 'disease', 'fungal', 'bacterial', 'viral', 'blight',
                    'rot', 'mold', 'mosaic', 'rust', 'spot'}
    
    for token in tokens:
        if token in crop_list:
            crops.append(token)
        elif token in nutrient_list:
            nutrients.append(token)
        elif token in fert_list:
            fertilizers.append(token)
        elif token in soil_list:
            soil_types.append(token)
        elif token in disease_list:
            diseases.append(token)
    
    return {
        'crops': list(set(crops)),
        'nutrients': list(set(nutrients)),
        'fertilizers': list(set(fertilizers)),
        'soil_types': list(set(soil_types)),
        'diseases': list(set(diseases)),
    }


def batch_preprocess_text(texts: List[str], **kwargs) -> List[str]:
    """
    Apply preprocessing to a batch of texts.
    
    Args:
        texts: List of input texts
        **kwargs: Arguments to pass to preprocess_text()
    
    Returns:
        List of preprocessed texts
    """
    return [preprocess_text(t, **kwargs) for t in texts]


def get_preprocessing_stats(original: str, processed: str) -> Dict:
    """
    Generate statistics about the preprocessing transformation.
    
    Args:
        original: Original text
        processed: Preprocessed text
    
    Returns:
        Dict with statistics
    """
    return {
        'original_length': len(original),
        'processed_length': len(processed),
        'original_tokens': len(original.split()),
        'processed_tokens': len(processed.split()),
        'chars_removed': len(original) - len(processed),
        'tokens_removed': len(original.split()) - len(processed.split()),
        'changed': original != processed,
    }


# ─────────────────────────────────────────────────────────────────
# EXAMPLE TEST CASES
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("TEXT PREPROCESSING PIPELINE - EXAMPLES")
    print("=" * 70)
    
    test_cases = [
        "What fertilizer should I use for my Phosphorous deficiency???",
        "My crop has pH 6.5 and N=90, P=42... what should I do?",
        "I want to grow rice in loamy soil with high humidity!",
        "There are brown spots on my wheat leaves, how do I treat them?",
        "Is it a good time to plant maize given current weather???",
        "My nitrogen is 70, phosphorous is 30, potassium is 50 - help!",
    ]
    
    for text in test_cases:
        print(f"\n📌 ORIGINAL:\n   {text}")
        processed = preprocess_text(text)
        print(f"✅ PROCESSED:\n   {processed}")
        
        # Extract numbers
        numbers = extract_numbers(text)
        if numbers['values']:
            print(f"📊 NUMBERS FOUND: {numbers['values']}")
        
        # Detect keywords
        keywords = detect_keywords(processed)
        detected = {k: v for k, v in keywords.items() if v}
        if detected:
            print(f"🔑 KEYWORDS: {detected}")
        
        # Stats
        stats = get_preprocessing_stats(text, processed)
        print(f"📈 STATS: {stats['original_tokens']} → {stats['processed_tokens']} tokens")
        print("-" * 70)
