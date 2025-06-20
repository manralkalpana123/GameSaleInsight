import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Any

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for video game sales data."""
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.cleaning_report = {}
    
    def load_data(self, uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV file."""
        try:
            # Read CSV file
            data = pd.read_csv(uploaded_file)
            self.original_data = data.copy()
            self.data = data.copy()
            return data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality and return a comprehensive report."""
        if self.data is None:
            return {}
        
        report = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'numeric_columns': self.data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Calculate missing percentage
        report['missing_percentage'] = {
            col: (missing / len(self.data)) * 100 
            for col, missing in report['missing_values'].items()
        }
        
        return report
    
    def clean_data(self, cleaning_options: Dict[str, Any]) -> pd.DataFrame:
        """Clean data based on provided options."""
        if self.data is None:
            return None
        
        cleaned_data = self.data.copy()
        self.cleaning_report = {'steps_applied': []}
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', False):
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            after_count = len(cleaned_data)
            self.cleaning_report['steps_applied'].append(
                f"Removed {before_count - after_count} duplicate rows"
            )
        
        # Handle missing values
        missing_strategy = cleaning_options.get('missing_strategy', 'drop')
        
        if missing_strategy == 'drop':
            before_count = len(cleaned_data)
            cleaned_data = cleaned_data.dropna()
            after_count = len(cleaned_data)
            self.cleaning_report['steps_applied'].append(
                f"Dropped {before_count - after_count} rows with missing values"
            )
        
        elif missing_strategy == 'fill_numeric':
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_data[col].isnull().any():
                    fill_value = cleaned_data[col].median()
                    cleaned_data[col].fillna(fill_value, inplace=True)
                    self.cleaning_report['steps_applied'].append(
                        f"Filled missing values in {col} with median: {fill_value:.2f}"
                    )
        
        elif missing_strategy == 'fill_categorical':
            categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if cleaned_data[col].isnull().any():
                    mode_value = cleaned_data[col].mode()[0] if not cleaned_data[col].mode().empty else 'Unknown'
                    cleaned_data[col].fillna(mode_value, inplace=True)
                    self.cleaning_report['steps_applied'].append(
                        f"Filled missing values in {col} with mode: {mode_value}"
                    )
        
        # Data type conversions
        if cleaning_options.get('convert_types', False):
            # Convert Year to numeric if it exists
            if 'Year' in cleaned_data.columns:
                try:
                    cleaned_data['Year'] = pd.to_numeric(cleaned_data['Year'], errors='coerce')
                    self.cleaning_report['steps_applied'].append("Converted Year to numeric")
                except:
                    pass
            
            # Convert sales columns to numeric
            sales_columns = [col for col in cleaned_data.columns if 'Sales' in col or 'sales' in col]
            for col in sales_columns:
                try:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                    self.cleaning_report['steps_applied'].append(f"Converted {col} to numeric")
                except:
                    pass
        
        # Remove outliers
        if cleaning_options.get('remove_outliers', False):
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_count = len(cleaned_data)
                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)
                ]
                after_count = len(cleaned_data)
                
                if before_count != after_count:
                    self.cleaning_report['steps_applied'].append(
                        f"Removed {before_count - after_count} outliers from {col}"
                    )
        
        # Standardize text columns
        if cleaning_options.get('standardize_text', False):
            text_cols = cleaned_data.select_dtypes(include=['object']).columns
            for col in text_cols:
                cleaned_data[col] = cleaned_data[col].str.strip().str.title()
                self.cleaning_report['steps_applied'].append(f"Standardized text in {col}")
        
        self.data = cleaned_data
        return cleaned_data
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Return the cleaning report."""
        return self.cleaning_report
    
    def export_data(self, data: pd.DataFrame, filename: str = "cleaned_data.csv") -> bytes:
        """Export cleaned data to CSV format."""
        return data.to_csv(index=False).encode('utf-8')
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get detailed information about each column."""
        if self.data is None:
            return {}
        
        column_info = {}
        for col in self.data.columns:
            column_info[col] = {
                'dtype': str(self.data[col].dtype),
                'non_null_count': self.data[col].count(),
                'null_count': self.data[col].isnull().sum(),
                'unique_values': self.data[col].nunique(),
                'memory_usage': self.data[col].memory_usage(deep=True)
            }
            
            if self.data[col].dtype in ['int64', 'float64']:
                column_info[col].update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'median': self.data[col].median(),
                    'std': self.data[col].std()
                })
            else:
                # For categorical columns, show top values
                value_counts = self.data[col].value_counts().head()
                column_info[col]['top_values'] = value_counts.to_dict()
        
        return column_info
