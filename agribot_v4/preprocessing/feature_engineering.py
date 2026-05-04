"""
preprocessing/feature_engineering.py

Feature engineering pipeline:
- Temperature range categorization
- Humidity level classification
- Soil-crop interaction features
- Feature normalization and scaling
- Derived feature creation
- Categorical encoding (One-Hot, Label Encoding)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Complete feature engineering pipeline for agricultural data."""
    
    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            df: Input DataFrame (should be cleaned first)
            verbose: Print operations
        """
        self.df = df.copy()
        self.original_features = list(df.columns)
        self.verbose = verbose
        self.scalers = {}
        self.encoders = {}
        self.engineering_log = []
    
    def _log(self, message: str):
        """Log feature engineering operations."""
        if self.verbose:
            print(f"  ✓ {message}")
        self.engineering_log.append(message)
    
    def categorize_temperature(self, col: str = 'temperature', new_col: str = 'temp_category') -> 'FeatureEngineer':
        """
        Categorize temperature into ranges:
        - Cold: < 15°C
        - Cool: 15-20°C
        - Moderate: 20-25°C
        - Warm: 25-30°C
        - Hot: > 30°C
        """
        def categorize(temp):
            if temp < 15:
                return 'cold'
            elif temp < 20:
                return 'cool'
            elif temp < 25:
                return 'moderate'
            elif temp < 30:
                return 'warm'
            else:
                return 'hot'
        
        if col in self.df.columns:
            self.df[new_col] = self.df[col].apply(categorize)
            self._log(f"Created temperature categories ({new_col})")
        
        return self
    
    def categorize_humidity(self, col: str = 'humidity', new_col: str = 'humidity_level') -> 'FeatureEngineer':
        """
        Categorize humidity into levels:
        - Very Low: < 30%
        - Low: 30-50%
        - Moderate: 50-70%
        - High: 70-85%
        - Very High: > 85%
        """
        def categorize(hum):
            if hum < 30:
                return 'very_low'
            elif hum < 50:
                return 'low'
            elif hum < 70:
                return 'moderate'
            elif hum < 85:
                return 'high'
            else:
                return 'very_high'
        
        if col in self.df.columns:
            self.df[new_col] = self.df[col].apply(categorize)
            self._log(f"Created humidity categories ({new_col})")
        
        return self
    
    def create_npk_balance_features(self) -> 'FeatureEngineer':
        """
        Create features representing NPK balance and total nutrients.
        
        New features:
        - total_nutrients: N + P + K
        - n_ratio: N / (N+P+K)
        - p_ratio: P / (N+P+K)
        - k_ratio: K / (N+P+K)
        - np_ratio: N / P
        - nk_ratio: N / K
        """
        n_col = p_col = k_col = None
        
        # Find nitrogen column (various naming conventions)
        for col in self.df.columns:
            if col in ['n', 'nitrogen', 'n_level']:
                n_col = col
                break
        
        # Find phosphorus column
        for col in self.df.columns:
            if col in ['p', 'phosphorus', 'phosphorous', 'p_level']:
                p_col = col
                break
        
        # Find potassium column
        for col in self.df.columns:
            if col in ['k', 'potassium', 'potassium_level']:
                k_col = col
                break
        
        if n_col and p_col and k_col:
            # Total nutrients
            self.df['total_nutrients'] = self.df[n_col] + self.df[p_col] + self.df[k_col]
            
            # Ratios (avoid division by zero)
            total = self.df['total_nutrients']
            self.df['n_ratio'] = np.divide(self.df[n_col], total, where=total != 0, out=np.zeros_like(total))
            self.df['p_ratio'] = np.divide(self.df[p_col], total, where=total != 0, out=np.zeros_like(total))
            self.df['k_ratio'] = np.divide(self.df[k_col], total, where=total != 0, out=np.zeros_like(total))
            
            # Pairwise ratios
            self.df['np_ratio'] = np.divide(self.df[n_col], self.df[p_col], 
                                           where=self.df[p_col] != 0, out=np.zeros_like(self.df[p_col]))
            self.df['nk_ratio'] = np.divide(self.df[n_col], self.df[k_col],
                                           where=self.df[k_col] != 0, out=np.zeros_like(self.df[k_col]))
            self.df['pk_ratio'] = np.divide(self.df[p_col], self.df[k_col],
                                           where=self.df[k_col] != 0, out=np.zeros_like(self.df[k_col]))
            
            self._log("Created NPK balance features (ratios and totals)")
        
        return self
    
    def create_soil_nutrient_level_features(self) -> 'FeatureEngineer':
        """
        Categorize soil nutrient levels:
        - Deficient: < 25th percentile
        - Low: 25-50th percentile
        - Moderate: 50-75th percentile
        - High: > 75th percentile
        """
        for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
            # Try to find the column
            col = None
            for c in self.df.columns:
                if nutrient.lower() in c.lower() or nutrient[0] == c.lower():
                    col = c
                    break
            
            if col is not None:
                q25 = self.df[col].quantile(0.25)
                q50 = self.df[col].quantile(0.50)
                q75 = self.df[col].quantile(0.75)
                
                def categorize(val):
                    if val < q25:
                        return 'deficient'
                    elif val < q50:
                        return 'low'
                    elif val < q75:
                        return 'moderate'
                    else:
                        return 'high'
                
                new_col = f"{nutrient[0]}_level"
                self.df[new_col] = self.df[col].apply(categorize)
        
        self._log("Created soil nutrient level categories")
        return self
    
    def create_rainfall_category(self, col: str = 'rainfall', new_col: str = 'rainfall_category') -> 'FeatureEngineer':
        """
        Categorize rainfall:
        - Very Low: < 500mm/year
        - Low: 500-1000mm
        - Moderate: 1000-1500mm
        - High: 1500-2500mm
        - Very High: > 2500mm
        """
        def categorize(rain):
            if rain < 500:
                return 'very_low'
            elif rain < 1000:
                return 'low'
            elif rain < 1500:
                return 'moderate'
            elif rain < 2500:
                return 'high'
            else:
                return 'very_high'
        
        if col in self.df.columns:
            self.df[new_col] = self.df[col].apply(categorize)
            self._log(f"Created rainfall categories ({new_col})")
        
        return self
    
    def create_ph_acidity_level(self, col: str = 'ph', new_col: str = 'ph_level') -> 'FeatureEngineer':
        """
        Categorize soil pH:
        - Highly Acidic: < 5.5
        - Acidic: 5.5-6.5
        - Neutral: 6.5-7.5
        - Alkaline: 7.5-8.5
        - Highly Alkaline: > 8.5
        """
        def categorize(ph):
            if ph < 5.5:
                return 'highly_acidic'
            elif ph < 6.5:
                return 'acidic'
            elif ph < 7.5:
                return 'neutral'
            elif ph < 8.5:
                return 'alkaline'
            else:
                return 'highly_alkaline'
        
        if col in self.df.columns:
            self.df[new_col] = self.df[col].apply(categorize)
            self._log(f"Created pH acidity levels ({new_col})")
        
        return self
    
    def encode_categorical_features(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'label',
        fit_encoders: bool = True
    ) -> 'FeatureEngineer':
        """
        Encode categorical features.
        
        Args:
            columns: Columns to encode (all categorical if None)
            method: 'label' or 'onehot'
            fit_encoders: Fit new encoders or use existing
        
        Returns:
            Self
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'label':
                if fit_encoders:
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.encoders[col] = le
                else:
                    if col in self.encoders:
                        self.df[f'{col}_encoded'] = self.encoders[col].transform(
                            self.df[col].astype(str))
                
            elif method == 'onehot':
                if fit_encoders:
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = ohe.fit_transform(self.df[[col]])
                    
                    # Create one-hot encoded columns
                    for i, category in enumerate(ohe.categories_[0]):
                        self.df[f'{col}_{category}'] = encoded[:, i]
                    
                    self.encoders[col] = ohe
                else:
                    if col in self.encoders:
                        encoded = self.encoders[col].transform(self.df[[col]])
                        for i, category in enumerate(self.encoders[col].categories_[0]):
                            self.df[f'{col}_{category}'] = encoded[:, i]
        
        method_str = 'label encoding' if method == 'label' else 'one-hot encoding'
        self._log(f"Applied {method_str} to {len(columns)} categorical columns")
        
        return self
    
    def scale_numeric_features(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'standard',
        fit_scaler: bool = True
    ) -> 'FeatureEngineer':
        """
        Scale numeric features.
        
        Args:
            columns: Columns to scale (all numeric if None)
            method: 'standard' (z-score) or 'minmax' (0-1)
            fit_scaler: Fit new scaler or use existing
        
        Returns:
            Self
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ratio/engineered features that are already normalized
            exclude = [c for c in columns if 'ratio' in c or 'level' in c]
            columns = [c for c in columns if c not in exclude]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit_scaler:
            self.df[columns] = scaler.fit_transform(self.df[columns])
            self.scalers['numeric'] = scaler
        else:
            if 'numeric' in self.scalers:
                self.df[columns] = self.scalers['numeric'].transform(self.df[columns])
        
        self._log(f"Applied {method} scaling to {len(columns)} numeric columns")
        
        return self
    
    def create_interaction_features(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> 'FeatureEngineer':
        """
        Create interaction features (multiply two features).
        
        Args:
            feature_pairs: List of (col1, col2) tuples to multiply
        
        Returns:
            Self
        """
        if feature_pairs is None:
            # Auto-create meaningful interactions
            feature_pairs = [
                ('temperature', 'humidity'),
                ('temperature', 'rainfall'),
                ('humidity', 'rainfall'),
            ]
            # Add NPK interactions if available
            n_col = next((c for c in self.df.columns if c.lower() in ['n', 'nitrogen']), None)
            p_col = next((c for c in self.df.columns if c.lower() in ['p', 'phosphorus']), None)
            if n_col and p_col:
                feature_pairs.append((n_col, p_col))
        
        for col1, col2 in feature_pairs:
            if col1 in self.df.columns and col2 in self.df.columns:
                new_col = f"{col1}_x_{col2}"
                self.df[new_col] = self.df[col1] * self.df[col2]
        
        self._log(f"Created {len(feature_pairs)} interaction features")
        
        return self
    
    def create_polynomial_features(
        self,
        columns: Optional[List[str]] = None,
        degree: int = 2
    ) -> 'FeatureEngineer':
        """
        Create polynomial features (squared terms).
        
        Args:
            columns: Columns to create polynomials for
            degree: Polynomial degree (typically 2 for squared)
        
        Returns:
            Self
        """
        if columns is None:
            # Auto-select important features
            columns = []
            for col in self.df.columns:
                if col.lower() in ['temperature', 'humidity', 'rainfall', 'ph']:
                    columns.append(col)
        
        count = 0
        for col in columns:
            if col in self.df.columns:
                for d in range(2, degree + 1):
                    new_col = f"{col}_^{d}"
                    self.df[new_col] = self.df[col] ** d
                    count += 1
        
        self._log(f"Created {count} polynomial features")
        
        return self
    
    def feature_statistics(self) -> Dict:
        """Get statistics on engineered features."""
        return {
            'original_features': len(self.original_features),
            'current_features': len(self.df.columns),
            'new_features_count': len(self.df.columns) - len(self.original_features),
            'new_features': [c for c in self.df.columns if c not in self.original_features],
        }
    
    def get_df(self) -> pd.DataFrame:
        """Return DataFrame with engineered features."""
        return self.df
    
    def get_scalers(self) -> Dict:
        """Return fitted scalers for later use."""
        return self.scalers
    
    def get_encoders(self) -> Dict:
        """Return fitted encoders for later use."""
        return self.encoders


def engineer_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_polynomial: bool = False,
    scale: bool = True,
    encode: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for feature engineering.
    
    Args:
        df: Input DataFrame (should be cleaned first)
        include_interactions: Create interaction features
        include_polynomial: Create polynomial features
        scale: Scale numeric features
        encode: Encode categorical features
        **kwargs: Additional arguments
    
    Returns:
        DataFrame with engineered features and scalers/encoders
    """
    engineer = FeatureEngineer(df, verbose=kwargs.get('verbose', True))
    
    # Create derived features
    engineer.categorize_temperature()
    engineer.categorize_humidity()
    engineer.create_npk_balance_features()
    engineer.create_soil_nutrient_level_features()
    engineer.create_rainfall_category()
    engineer.create_ph_acidity_level()
    
    # Encode categorical features
    if encode:
        engineer.encode_categorical_features(method='label')
    
    # Create interaction features
    if include_interactions:
        engineer.create_interaction_features()
    
    # Create polynomial features
    if include_polynomial:
        engineer.create_polynomial_features(degree=2)
    
    # Scale numeric features
    if scale:
        engineer.scale_numeric_features(method='standard')
    
    stats = engineer.feature_statistics()
    
    return engineer.get_df(), {
        'scalers': engineer.get_scalers(),
        'encoders': engineer.get_encoders(),
        'statistics': stats,
    }


# ─────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE - EXAMPLES")
    print("=" * 70)
    
    # Create sample dataset
    sample_data = {
        'N': [90, 85, 60, 74, 78],
        'P': [42, 58, 55, 35, 42],
        'K': [43, 41, 44, 40, 42],
        'temperature': [20.9, 21.8, 23.0, 26.5, 20.9],
        'humidity': [82.0, 80.3, 82.3, 80.2, 82.0],
        'rainfall': [202.9, 226.7, 263.9, 242.9, 262.7],
        'ph': [6.5, 7.0, 7.8, 6.9, 6.3],
        'crop': ['rice', 'rice', 'rice', 'rice', 'wheat'],
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\n📊 ORIGINAL FEATURES:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    print("\n" + "=" * 70)
    print("APPLYING FEATURE ENGINEERING")
    print("=" * 70)
    
    # Engineer features
    engineered_df, artifacts = engineer_features(
        df,
        include_interactions=True,
        include_polynomial=False,
        scale=True,
        encode=True,
        verbose=True
    )
    
    print("\n📊 ENGINEERED FEATURES:")
    stats = artifacts['statistics']
    print(f"Original features: {stats['original_features']}")
    print(f"Current features: {stats['current_features']}")
    print(f"New features created: {stats['new_features_count']}")
    print(f"\nNew features: {stats['new_features']}")
    
    print("\n📋 ENGINEERED DATAFRAME (first 3 rows):")
    print(engineered_df.head(3).to_string())
    
    print("\n✅ Feature engineering complete!")
