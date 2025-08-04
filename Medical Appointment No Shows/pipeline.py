import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

# Custom transformer classes for each cleaning step
class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, rename_dict=None, make_lowercase=True, replace_spaces=True):
        self.rename_dict = rename_dict
        self.make_lowercase = make_lowercase
        self.replace_spaces = replace_spaces
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # Apply custom renaming if provided
        if self.rename_dict:
            X_copy = X_copy.rename(columns=self.rename_dict)
        
        # Make all column names lowercase
        if self.make_lowercase:
            X_copy.columns = [col.lower() for col in X_copy.columns]
        
        # Replace spaces with underscores
        if self.replace_spaces:
            X_copy.columns = [col.replace(' ', '_') for col in X_copy.columns]
            
        return X_copy

class DuplicateHandler(BaseEstimator, TransformerMixin):
    def __init__(self, subset=None):
        self.subset = subset
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        before_count = X.shape[0]
        X_copy = X.drop_duplicates(subset=self.subset)
        after_count = X_copy.shape[0]
        print(f"Removed {before_count - after_count} duplicate rows")
        return X_copy

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent', 
                 drop_cols_threshold=0.7, custom_fill=None):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.drop_cols_threshold = drop_cols_threshold
        self.custom_fill = custom_fill
        self.fill_values = {}
        
    def fit(self, X, y=None):
        # Calculate fill values for numeric columns
        for col in X.select_dtypes(include=['number']).columns:
            if self.numeric_strategy == 'mean':
                self.fill_values[col] = X[col].mean()
            elif self.numeric_strategy == 'median':
                self.fill_values[col] = X[col].median()
            elif self.numeric_strategy == 'zero':
                self.fill_values[col] = 0
                
        # Calculate fill values for categorical columns
        for col in X.select_dtypes(exclude=['number']).columns:
            if self.categorical_strategy == 'most_frequent':
                self.fill_values[col] = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
            elif self.categorical_strategy == 'missing':
                self.fill_values[col] = "Missing"
                
        # Override with custom fill values if provided
        if self.custom_fill:
            self.fill_values.update(self.custom_fill)
            
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Drop columns with too many missing values
        if self.drop_cols_threshold:
            missing_percent = X_copy.isnull().mean()
            cols_to_drop = missing_percent[missing_percent > self.drop_cols_threshold].index
            X_copy = X_copy.drop(columns=cols_to_drop)
            if len(cols_to_drop) > 0:
                print(f"Dropped columns with >={self.drop_cols_threshold*100}% missing values: {list(cols_to_drop)}")
        
        # Fill missing values
        for col in X_copy.columns:
            if col in self.fill_values:
                X_copy[col] = X_copy[col].fillna(self.fill_values[col])
                
        return X_copy

class DateFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns=None, date_format='%Y-%m-%d'):
        self.date_columns = date_columns
        self.date_format = date_format
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.date_columns:
            for col in self.date_columns:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')
                    X_copy[col] = X_copy[col].dt.strftime(self.date_format)
                    
        return X_copy

class DataTypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, type_conversions=None):
        self.type_conversions = type_conversions
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.type_conversions:
            for col, dtype in self.type_conversions.items():
                if col in X_copy.columns:
                    try:
                        X_copy[col] = X_copy[col].astype(dtype)
                    except:
                        print(f"Could not convert {col} to {dtype}")
                        
        return X_copy

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, method='iqr', threshold=1.5):
        self.columns = columns
        self.method = method
        self.threshold = threshold
        self.limits = {}
        
    def fit(self, X, y=None):
        if not self.columns:
            self.columns = X.select_dtypes(include=['number']).columns
            
        for col in self.columns:
            if col in X.columns:
                if self.method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.limits[col] = {
                        'lower': Q1 - self.threshold * IQR,
                        'upper': Q3 + self.threshold * IQR
                    }
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    self.limits[col] = {
                        'lower': mean - self.threshold * std,
                        'upper': mean + self.threshold * std
                    }
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col, limits in self.limits.items():
            if col in X_copy.columns:
                # Cap outliers at the defined limits
                outliers_count = ((X_copy[col] < limits['lower']) | (X_copy[col] > limits['upper'])).sum()
                if outliers_count > 0:
                    print(f"Capping {outliers_count} outliers in column '{col}'")
                    X_copy[col] = X_copy[col].clip(lower=limits['lower'], upper=limits['upper'])
                    
        return X_copy

class TextStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, case='lower', remove_special_chars=True):
        self.columns = columns
        self.case = case
        self.remove_special_chars = remove_special_chars
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if not self.columns:
            self.columns = X_copy.select_dtypes(include=['object']).columns
            
        for col in self.columns:
            if col in X_copy.columns:
                # Convert case if specified
                if self.case == 'lower':
                    X_copy[col] = X_copy[col].str.lower()
                elif self.case == 'upper':
                    X_copy[col] = X_copy[col].str.upper()
                elif self.case == 'title':
                    X_copy[col] = X_copy[col].str.title()
                
                # Remove special characters if specified
                if self.remove_special_chars:
                    X_copy[col] = X_copy[col].str.replace(r'[^\w\s]', '', regex=True)
                
                # Strip whitespace
                X_copy[col] = X_copy[col].str.strip()
                
        return X_copy

class CategoryStandardizer(BaseEstimator, TransformerMixin):
    def __init__(self, mappings=None):
        self.mappings = mappings
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.mappings:
            for col, mapping in self.mappings.items():
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].map(mapping).fillna(X_copy[col])
                    
        return X_copy

# Create and use the pipeline
def create_data_cleaning_pipeline(
    rename_columns=None,
    date_columns=None,
    type_conversions=None,
    category_mappings=None,
    numeric_missing_strategy='mean',
    categorical_missing_strategy='most_frequent',
    custom_fill_values=None,
    drop_threshold=0.7
):
    """
    Create a customizable data cleaning pipeline
    
    Parameters:
    -----------
    rename_columns : dict, optional
        Dictionary mapping original column names to new names
    date_columns : list, optional
        List of columns to convert to datetime format
    type_conversions : dict, optional
        Dictionary mapping column names to desired data types
    category_mappings : dict, optional
        Dictionary of dictionaries for standardizing categorical values
        Example: {'gender': {'m': 'Male', 'f': 'Female'}}
    numeric_missing_strategy : str, optional
        Strategy for filling numeric missing values ('mean', 'median', 'zero')
    categorical_missing_strategy : str, optional
        Strategy for filling categorical missing values ('most_frequent', 'missing')
    custom_fill_values : dict, optional
        Dictionary mapping column names to specific fill values
    drop_threshold : float, optional
        Drop columns with more than this proportion of missing values (0 to 1)
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        A data cleaning pipeline
    """
    
    pipeline = Pipeline([
        ('rename_columns', ColumnRenamer(rename_dict=rename_columns)),
        ('handle_duplicates', DuplicateHandler()),
        ('handle_missing_values', MissingValueHandler(
            numeric_strategy=numeric_missing_strategy,
            categorical_strategy=categorical_missing_strategy,
            drop_cols_threshold=drop_threshold,
            custom_fill=custom_fill_values
        )),
        ('standardize_categories', CategoryStandardizer(mappings=category_mappings)),
        ('format_dates', DateFormatter(date_columns=date_columns)),
        ('convert_types', DataTypeConverter(type_conversions=type_conversions)),
        ('handle_outliers', OutlierHandler()),
        ('standardize_text', TextStandardizer())
    ])
    
    return pipeline

# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')
    
    # Define your cleaning parameters
    params = {
        'rename_columns': {'Customer ID': 'customer_id', 'Purchase Date': 'purchase_date'},
        'date_columns': ['purchase_date', 'delivery_date'],
        'type_conversions': {'age': 'int', 'customer_id': 'str'},
        'category_mappings': {
            'gender': {'m': 'Male', 'f': 'Female', 'male': 'Male', 'female': 'Female'},
            'country': {'usa': 'United States', 'uk': 'United Kingdom'}
        },
        'custom_fill_values': {'age': 30, 'income': 50000}
    }
    
    # Create and run the pipeline
    cleaning_pipeline = create_data_cleaning_pipeline(**params)
    df_cleaned = cleaning_pipeline.fit_transform(df)
    
    # Generate summary of changes
    print("\n=== Data Cleaning Summary ===")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {df.shape[0] - df_cleaned.shape[0]}")
    print(f"Columns removed: {df.shape[1] - df_cleaned.shape[1]}")
    
    # Check for any remaining missing values
    missing = df_cleaned.isnull().sum()
    print("\nRemaining missing values:")
    print(missing[missing > 0] if missing.any() > 0 else "None")
    
    # Save the cleaned dataset
    df_cleaned.to_csv('cleaned_dataset.csv', index=False)
    print("\nCleaned dataset saved to 'cleaned_dataset.csv'")