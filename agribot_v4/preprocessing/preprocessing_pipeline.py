"""
preprocessing/preprocessing_pipeline.py

Master preprocessing pipeline that orchestrates:
1. Text preprocessing (for user queries)
2. Data cleaning (missing values, duplicates, outliers)
3. Feature engineering (derived features, encoding, scaling)

This module provides a unified interface for complete data preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import warnings

from .text_preprocessing import preprocess_text, batch_preprocess_text, extract_numbers, detect_keywords
from .data_cleaning import DataCleaner, clean_dataset
from .feature_engineering import FeatureEngineer, engineer_features

warnings.filterwarnings("ignore")


class ComprehensivePreprocessingPipeline:
    """
    Complete preprocessing pipeline for agricultural ML models.
    
    Handles:
    - Text input preprocessing for queries
    - CSV dataset cleaning and validation
    - Feature engineering and encoding
    - Model-ready data preparation
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize pipeline.
        
        Args:
            verbose: Print operations
        """
        self.verbose = verbose
        self.pipeline_log = []
        self.artifacts = {}
    
    def _log(self, message: str, level: str = "info"):
        """Log pipeline operations."""
        prefix = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
        }.get(level, "→ ")
        
        if self.verbose:
            print(f"{prefix} {message}")
        self.pipeline_log.append({"message": message, "level": level})
    
    # ── TEXT PREPROCESSING ─────────────────────────────────────────────────
    
    def preprocess_user_query(self, text: str) -> Dict[str, Any]:
        """
        Preprocess a user query for intent classification.
        
        Args:
            text: Raw user input
        
        Returns:
            Dict with preprocessed text, tokens, extracted numbers, and keywords
        """
        self._log(f"Preprocessing user query: '{text[:50]}...'")
        
        # Clean text
        cleaned = preprocess_text(text, lowercase=True, remove_punct=True)
        
        # Extract numbers
        numbers = extract_numbers(text)
        
        # Detect keywords
        keywords = detect_keywords(cleaned)
        
        # Tokenize
        tokens = cleaned.split()
        
        result = {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'numbers': numbers,
            'keywords': keywords,
            'token_count': len(tokens),
        }
        
        self._log(f"Query preprocessed: {len(tokens)} tokens, {len(keywords)} keyword categories")
        
        return result
    
    def preprocess_query_batch(self, texts: List[str]) -> List[Dict]:
        """Preprocess multiple queries."""
        results = []
        for text in texts:
            results.append(self.preprocess_user_query(text))
        
        self._log(f"Batch preprocessed {len(texts)} queries")
        return results
    
    # ── DATA CLEANING ─────────────────────────────────────────────────────
    
    def clean_csv_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Unknown",
        **cleaning_kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean a CSV dataset comprehensively.
        
        Args:
            df: Input DataFrame
            dataset_name: Name for logging
            **cleaning_kwargs: Arguments for DataCleaner
        
        Returns:
            Cleaned DataFrame and cleaning report
        """
        self._log(f"Starting data cleaning: {dataset_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
        
        cleaner = DataCleaner(df, verbose=self.verbose)
        cleaner.clean_all()
        
        cleaned_df = cleaner.get_df()
        report = cleaner.report()
        
        self._log(f"Cleaning complete: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} cols")
        self.artifacts[f'{dataset_name}_cleaning_report'] = report
        
        return cleaned_df, report
    
    # ── FEATURE ENGINEERING ────────────────────────────────────────────────
    
    def engineer_features_for_model(
        self,
        df: pd.DataFrame,
        model_type: str = "crop",
        **engineering_kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Engineer features for specific ML model.
        
        Args:
            df: Cleaned DataFrame
            model_type: 'crop', 'fertilizer', 'yield', etc.
            **engineering_kwargs: Feature engineering options
        
        Returns:
            DataFrame with engineered features and artifacts
        """
        self._log(f"Starting feature engineering for {model_type} model")
        
        engineered_df, artifacts = engineer_features(
            df,
            include_interactions=engineering_kwargs.get('include_interactions', True),
            include_polynomial=engineering_kwargs.get('include_polynomial', False),
            scale=engineering_kwargs.get('scale', True),
            encode=engineering_kwargs.get('encode', True),
            verbose=self.verbose
        )
        
        stats = artifacts['statistics']
        self._log(f"Feature engineering complete: {stats['current_features']} total features")
        self.artifacts[f'{model_type}_feature_artifacts'] = artifacts
        
        return engineered_df, artifacts
    
    # ── FULL PIPELINE ──────────────────────────────────────────────────────
    
    def process_dataset_full_pipeline(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        model_type: str = "general",
        skip_feature_engineering: bool = False,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply complete preprocessing pipeline to dataset.
        
        Steps:
        1. Data cleaning
        2. Feature engineering
        3. Final validation
        
        Args:
            df: Raw DataFrame
            dataset_name: Name for logging
            model_type: Type of ML model this is for
            skip_feature_engineering: Skip feature engineering step
        
        Returns:
            Fully preprocessed DataFrame and complete report
        """
        self._log("=" * 70)
        self._log(f"STARTING FULL PREPROCESSING PIPELINE: {dataset_name}")
        self._log("=" * 70)
        
        # Step 1: Data Cleaning
        cleaned_df, cleaning_report = self.clean_csv_dataset(df, dataset_name)
        
        # Step 2: Feature Engineering
        if not skip_feature_engineering:
            engineered_df, feature_artifacts = self.engineer_features_for_model(
                cleaned_df,
                model_type=model_type
            )
        else:
            engineered_df = cleaned_df
            feature_artifacts = None
        
        # Step 3: Final Validation
        final_shape = engineered_df.shape
        has_nan = engineered_df.isnull().sum().sum()
        has_inf = np.isinf(engineered_df.select_dtypes(include=[np.number])).sum().sum()
        
        self._log(f"Final validation: {final_shape[0]} rows, {final_shape[1]} cols")
        if has_nan > 0:
            self._log(f"Warning: {has_nan} NaN values remain", level="warning")
        if has_inf > 0:
            self._log(f"Warning: {has_inf} infinite values found", level="warning")
        
        self._log("=" * 70)
        self._log("PREPROCESSING PIPELINE COMPLETE")
        self._log("=" * 70)
        
        complete_report = {
            'dataset_name': dataset_name,
            'model_type': model_type,
            'original_shape': df.shape,
            'final_shape': final_shape,
            'cleaning_report': cleaning_report,
            'feature_artifacts': feature_artifacts,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'pipeline_log': self.pipeline_log,
        }
        
        return engineered_df, complete_report
    
    # ── DATASET-SPECIFIC PIPELINES ────────────────────────────────────────
    
    def process_crop_recommendation_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Specialized pipeline for crop recommendation dataset.
        
        Expected columns: N, P, K, temperature, humidity, ph, rainfall, label
        """
        return self.process_dataset_full_pipeline(
            df,
            dataset_name="crop_recommendation",
            model_type="crop",
        )
    
    def process_fertilizer_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Specialized pipeline for fertilizer recommendation dataset.
        
        Expected columns: temperature, humidity, moisture, soil_type, crop_type,
                         nitrogen, phosphorous, potassium, fertilizer
        """
        return self.process_dataset_full_pipeline(
            df,
            dataset_name="fertilizer",
            model_type="fertilizer",
        )
    
    def process_yield_prediction_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Specialized pipeline for yield prediction dataset.
        """
        return self.process_dataset_full_pipeline(
            df,
            dataset_name="yield",
            model_type="yield",
        )
    
    def get_artifacts(self) -> Dict:
        """Return all preprocessing artifacts (scalers, encoders, etc.)."""
        return self.artifacts
    
    def get_log(self) -> List[Dict]:
        """Return complete pipeline log."""
        return self.pipeline_log


# ─────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def preprocess_for_model(
    csv_path: str,
    model_type: str = "crop",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and preprocess CSV dataset for a specific ML model.
    
    Args:
        csv_path: Path to CSV file
        model_type: 'crop', 'fertilizer', 'yield'
        verbose: Print operations
    
    Returns:
        Preprocessed DataFrame and report
    """
    df = pd.read_csv(csv_path)
    
    pipeline = ComprehensivePreprocessingPipeline(verbose=verbose)
    
    if model_type == "crop":
        return pipeline.process_crop_recommendation_dataset(df)
    elif model_type == "fertilizer":
        return pipeline.process_fertilizer_dataset(df)
    elif model_type == "yield":
        return pipeline.process_yield_prediction_dataset(df)
    else:
        return pipeline.process_dataset_full_pipeline(df, dataset_name=csv_path, model_type=model_type)


# ─────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PREPROCESSING PIPELINE - EXAMPLES")
    print("=" * 70)
    
    # Example 1: Query preprocessing
    print("\n📝 EXAMPLE 1: USER QUERY PREPROCESSING")
    print("-" * 70)
    
    pipeline = ComprehensivePreprocessingPipeline(verbose=True)
    
    queries = [
        "What crop should I grow with N=90, P=42, K=43?",
        "My soil has phosphorous deficiency, what fertilizer should I use???",
        "Is it a good time to plant rice in loamy soil?",
    ]
    
    for query in queries:
        result = pipeline.preprocess_user_query(query)
        print(f"\n  Original: {result['original']}")
        print(f"  Cleaned:  {result['cleaned']}")
        print(f"  Tokens:   {result['tokens']}")
        print(f"  Numbers:  {result['numbers']['values']}")
        print(f"  Keywords: {result['keywords']}")
    
    # Example 2: Dataset preprocessing
    print("\n\n📊 EXAMPLE 2: CSV DATASET PREPROCESSING")
    print("-" * 70)
    
    # Create sample dataset
    sample_data = {
        'N': [90, 85, None, 120, 90],
        'P': [42, 58, 55, 40, 42],
        'K': [43, 41, 44, None, 43],
        'temperature': [20.9, 21.8, 23.0, 26.5, 20.9],
        'humidity': [82.0, 80.3, 82.3, 150.0, 82.0],
        'ph': [6.5, 7.0, 7.8, 6.9, 6.3],
        'rainfall': [202.9, 226.7, 263.9, 242.9, 262.7],
        'label': ['rice', 'rice', 'rice', 'rice', 'wheat'],
    }
    
    df = pd.DataFrame(sample_data)
    print("\nOriginal dataset:")
    print(df)
    
    # Preprocess
    pipeline2 = ComprehensivePreprocessingPipeline(verbose=True)
    processed_df, report = pipeline2.process_crop_recommendation_dataset(df)
    
    print("\nPreprocessed dataset:")
    print(processed_df)
    
    print("\nPreprocessing report:")
    print(f"  Original shape: {report['original_shape']}")
    print(f"  Final shape: {report['final_shape']}")
    print(f"  Rows removed: {report['original_shape'][0] - report['final_shape'][0]}")
    print(f"  Features engineered: {report['feature_artifacts']['statistics']['new_features_count']}")
