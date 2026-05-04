"""
preprocessing/data_cleaning.py

Comprehensive data cleaning pipeline:
- Missing value handling
- Duplicate removal
- Outlier detection and handling
- Column name normalization
- Data type validation
- Inconsistent data fixing
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class DataCleaner:
    """Comprehensive data cleaning utilities for agricultural datasets."""
    
    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        """
        Initialize the DataCleaner.
        
        Args:
            df: Input DataFrame
            verbose: Print cleaning operations
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.verbose = verbose
        self.cleaning_log = []
    
    def _log(self, message: str):
        """Log cleaning operations."""
        if self.verbose:
            print(f"  ✓ {message}")
        self.cleaning_log.append(message)
    
    def report(self) -> Dict:
        """Get cleaning report."""
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'cleaning_steps': self.cleaning_log,
        }
    
    def normalize_column_names(self) -> 'DataCleaner':
        """
        Normalize column names:
        - Convert to lowercase
        - Replace spaces with underscores
        - Remove special characters
        """
        original_cols = list(self.df.columns)
        
        new_cols = []
        for col in original_cols:
            # Lowercase
            col = col.lower()
            # Replace spaces with underscore
            col = col.replace(' ', '_')
            # Remove special characters except underscore
            col = ''.join(c if c.isalnum() or c == '_' else '' for c in col)
            new_cols.append(col)
        
        self.df.columns = new_cols
        
        if original_cols != new_cols:
            self._log(f"Normalized column names: {list(zip(original_cols, new_cols))[:3]}...")
        
        return self
    
    def handle_missing_values(
        self,
        strategy: str = 'drop',
        threshold: float = 0.5,
        numeric_fill: str = 'mean',
        categorical_fill: str = 'mode'
    ) -> 'DataCleaner':
        """
        Handle missing values intelligently.
        
        Args:
            strategy: 'drop' (rows), 'fill' (values)
            threshold: Drop column if missing > threshold
            numeric_fill: 'mean', 'median', or numeric value
            categorical_fill: 'mode' or specific value
        
        Returns:
            Self (for chaining)
        """
        initial_rows = len(self.df)
        
        # Step 1: Drop columns with too many missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self._log(f"Dropped {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing")
        
        # Step 2: Handle remaining missing values
        if strategy == 'drop':
            self.df = self.df.dropna()
            rows_removed = initial_rows - len(self.df)
            self._log(f"Dropped {rows_removed} rows with missing values")
        
        elif strategy == 'fill':
            # Fill numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    if numeric_fill == 'mean':
                        fill_val = self.df[col].mean()
                    elif numeric_fill == 'median':
                        fill_val = self.df[col].median()
                    else:
                        fill_val = float(numeric_fill)
                    self.df[col].fillna(fill_val, inplace=True)
            
            # Fill categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    if categorical_fill == 'mode':
                        fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    else:
                        fill_val = categorical_fill
                    self.df[col].fillna(fill_val, inplace=True)
            
            self._log(f"Filled missing values (numeric:{numeric_fill}, categorical:{categorical_fill})")
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows.
        
        Args:
            subset: Columns to consider for duplicates
            keep: 'first', 'last', or False (remove all duplicates)
        
        Returns:
            Self (for chaining)
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        rows_removed = initial_rows - len(self.df)
        
        if rows_removed > 0:
            self._log(f"Removed {rows_removed} duplicate rows")
        
        return self
    
    def detect_outliers(
        self,
        method: str = 'iqr',
        columns: Optional[List[str]] = None,
        threshold: float = 1.5
    ) -> Dict:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            method: 'iqr' (Interquartile Range) or 'zscore'
            columns: Numeric columns to check (all if None)
            threshold: IQR multiplier (1.5) or Z-score threshold (3)
        
        Returns:
            Dict with outlier information
        """
        numeric_cols = columns or self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
                outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                if len(outliers) > 0:
                    outlier_info[col] = {
                        'count': len(outliers),
                        'indices': outliers.index.tolist()[:5],  # First 5
                        'bounds': (lower, upper),
                    }
        
        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_mask = z_scores > threshold
                outliers = self.df[outliers_mask]
                
                if len(outliers) > 0:
                    outlier_info[col] = {
                        'count': len(outliers),
                        'indices': outliers.index.tolist()[:5],
                    }
        
        return outlier_info
    
    def handle_outliers(
        self,
        method: str = 'clip',
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> 'DataCleaner':
        """
        Handle outliers by clipping or removing.
        
        Args:
            method: 'clip' (limit to bounds) or 'remove' (delete rows)
            threshold: IQR multiplier
            columns: Columns to process
        
        Returns:
            Self (for chaining)
        """
        numeric_cols = columns or self.df.select_dtypes(include=[np.number]).columns
        initial_rows = len(self.df)
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            if method == 'clip':
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            elif method == 'remove':
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        
        rows_removed = initial_rows - len(self.df)
        if rows_removed > 0:
            self._log(f"Removed {rows_removed} rows with outliers (method={method})")
        
        return self
    
    def fix_data_types(self, type_mapping: Optional[Dict[str, str]] = None) -> 'DataCleaner':
        """
        Fix incorrect data types.
        
        Args:
            type_mapping: Dict mapping column names to target types
        
        Returns:
            Self (for chaining)
        """
        if type_mapping is None:
            # Auto-detect and fix
            for col in self.df.columns:
                if col in ['temperature', 'humidity', 'rainfall', 'ph', 'moisture',
                          'nitrogen', 'phosphorus', 'phosphorous', 'potassium']:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        else:
            for col, dtype in type_mapping.items():
                if col in self.df.columns:
                    if dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    elif dtype == 'string':
                        self.df[col] = self.df[col].astype('string')
        
        self._log("Fixed data types")
        return self
    
    def standardize_categorical_values(self) -> 'DataCleaner':
        """
        Standardize categorical values:
        - Lowercase
        - Strip whitespace
        - Unify variations
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Lowercase and strip
            self.df[col] = self.df[col].str.lower().str.strip()
            
            # Fix common variations (e.g., crop names)
            if col in ['crop', 'label', 'crop_type', 'crop_name']:
                replacements = {
                    'arhar/tur': 'pigeonpeas',
                    'bajra': 'millet',
                    'gram': 'chickpea',
                    'moong': 'mungbean',
                    'urad': 'blackgram',
                    'soyabean': 'soybean',
                    'rapeseed': 'mustard',
                    'cotton(lint)': 'cotton',
                }
                for old, new in replacements.items():
                    self.df[col] = self.df[col].str.replace(old, new, regex=False)
        
        self._log("Standardized categorical values")
        return self
    
    def validate_numeric_ranges(self, ranges: Optional[Dict[str, Tuple]] = None) -> Dict:
        """
        Validate that numeric values are within expected ranges.
        
        Args:
            ranges: Dict mapping column names to (min, max) tuples
        
        Returns:
            Dict with validation results
        """
        if ranges is None:
            ranges = {
                'temperature': (-10, 50),
                'humidity': (0, 100),
                'rainfall': (0, 10000),
                'ph': (0, 14),
                'nitrogen': (0, 200),
                'phosphorus': (0, 150),
                'phosphorous': (0, 150),
                'potassium': (0, 200),
                'moisture': (0, 100),
            }
        
        violations = {}
        for col, (min_val, max_val) in ranges.items():
            if col not in self.df.columns:
                continue
            
            invalid = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
            if len(invalid) > 0:
                violations[col] = {
                    'expected_range': (min_val, max_val),
                    'count': len(invalid),
                    'examples': invalid[col].head(3).tolist(),
                }
        
        return violations
    
    def clean_all(
        self,
        handle_missing: bool = True,
        remove_dups: bool = True,
        handle_outliers_flag: bool = True,
        standardize: bool = True,
        validate: bool = True,
    ) -> 'DataCleaner':
        """
        Apply full cleaning pipeline in sequence.
        
        Returns:
            Self (for chaining)
        """
        print(f"🔧 Starting data cleaning ({self.df.shape[0]} rows, {self.df.shape[1]} cols)")
        
        self.normalize_column_names()
        
        if handle_missing:
            self.handle_missing_values(strategy='drop')
        
        if remove_dups:
            self.remove_duplicates()
        
        if standardize:
            self.fix_data_types()
            self.standardize_categorical_values()
        
        if handle_outliers_flag:
            self.handle_outliers(method='clip')
        
        if validate:
            violations = self.validate_numeric_ranges()
            if violations:
                print(f"\n⚠️  Validation warnings: {len(violations)} columns")
        
        print(f"✅ Cleaning complete ({self.df.shape[0]} rows, {self.df.shape[1]} cols)")
        return self
    
    def get_df(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        return self.df


def clean_dataset(
    df: pd.DataFrame,
    full_pipeline: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for cleaning a dataset.
    
    Args:
        df: Input DataFrame
        full_pipeline: Use all cleaning steps
        **kwargs: Additional arguments
    
    Returns:
        Cleaned DataFrame and report
    """
    cleaner = DataCleaner(df, verbose=kwargs.get('verbose', True))
    
    if full_pipeline:
        cleaner.clean_all()
    
    return cleaner.get_df(), cleaner.report()


# ─────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DATA CLEANING PIPELINE - EXAMPLES")
    print("=" * 70)
    
    # Create sample dataset with issues
    sample_data = {
        'N': [90, 85, None, 120, 90, -50, 85],  # Missing and outlier
        'P': [42, 58, 55, None, 42, 42, 58],
        'K': [43, 41, 44, 40, 43, 43, 41],
        'temperature': [20.9, 21.8, 23.0, 26.5, 20.9, 20.9, 25.0],
        'humidity': [82.0, 80.3, 82.3, 150.0, 82.0, 82.0, 81.0],  # Outlier
        'crop': ['Rice', 'RICE', ' rice', 'Rice', 'WHEAT', 'WHEAT', 'wheat'],  # Inconsistent
        'label': ['rice', 'rice', 'rice', 'rice', 'rice', 'rice', 'rice'],
    }
    
    df = pd.DataFrame(sample_data)
    
    print("\n📊 ORIGINAL DATASET:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    print("\n" + "=" * 70)
    print("APPLYING DATA CLEANING PIPELINE")
    print("=" * 70)
    
    # Clean the data
    cleaned_df, report = clean_dataset(df, full_pipeline=True, verbose=True)
    
    print("\n📊 CLEANED DATASET:")
    print(cleaned_df)
    print(f"\nShape: {cleaned_df.shape}")
    
    print("\n📋 CLEANING REPORT:")
    for key, value in report.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")
