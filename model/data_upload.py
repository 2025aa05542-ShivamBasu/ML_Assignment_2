"""
Data Upload and Loading Module
Handles data loading, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Tuple, Optional


class DataUploader:
    """Class to handle data upload and preprocessing"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataUploader
        
        Args:
            data_dir (str): Directory to store uploaded files
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load CSV file into DataFrame
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe or None if error
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Successfully loaded: {file_path}")
            print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"✗ Error loading file: {str(e)}")
            return None
    
    def load_from_upload(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from Streamlit file uploader
        
        Args:
            uploaded_file: File from st.file_uploader()
            
        Returns:
            pd.DataFrame: Loaded dataframe or None if error
        """
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            print(f"✗ Error loading uploaded file: {str(e)}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate data quality
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        if df.shape[0] == 0:
            return False, "No rows in dataset"
        
        if df.shape[1] == 0:
            return False, "No columns in dataset"
        
        return True, "Data validation passed"
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get detailed information about the dataset
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            
        Returns:
            dict: Dictionary with data information
        """
        info = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
        }
        return info
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            df (pd.DataFrame): DataFrame with possible missing values
            strategy (str): Strategy to handle missing values
                           Options: 'drop', 'mean', 'median', 'forward_fill'
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        if strategy == "drop":
            df_cleaned = df.dropna()
            print(f"Dropped {len(df) - len(df_cleaned)} rows with missing values")
            return df_cleaned
        
        elif strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df
        
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            return df
        
        elif strategy == "forward_fill":
            df = df.fillna(method='ffill')
            return df
        
        else:
            print(f"Unknown strategy: {strategy}")
            return df
    
    def save_data(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Save DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"✓ Data saved to: {filepath}")
            return True
        except Exception as e:
            print(f"✗ Error saving data: {str(e)}")
            return False
    
    def get_saved_files(self) -> list:
        """
        Get list of saved data files
        
        Returns:
            list: List of CSV files in data directory
        """
        try:
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            return files
        except Exception as e:
            print(f"✗ Error getting saved files: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    uploader = DataUploader()
    
    # Example: Load a CSV file
    # df = uploader.load_csv("sample_data.csv")
    # if df is not None:
    #     is_valid, msg = uploader.validate_data(df)
    #     print(msg)
    #     info = uploader.get_data_info(df)
    #     print(info)
