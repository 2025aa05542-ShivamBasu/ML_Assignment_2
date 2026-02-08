"""
Obesity Dataset Loader
Specialized module for loading and processing the Obesity dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional
import os


class ObesityDatasetLoader:
    """Class to handle Obesity dataset loading and preprocessing"""
    
    # Default path to the dataset
    DEFAULT_PATH = r"D:\LocalGoogleDriveSync\MTech_BITS_PILANI_AI_ML\01_Semester_1\02_ML\00_Assignment\00_Assignment_2\Data\archive\ObesityDataSet_raw_and_data_sinthetic.csv"
    
    def __init__(self, file_path: str = None):
        """
        Initialize ObesityDatasetLoader
        
        Args:
            file_path (str): Path to the obesity dataset CSV file
        """
        if file_path is None:
            file_path = self.DEFAULT_PATH
        
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = "NObeyesdad"
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load the obesity dataset
        
        Returns:
            pd.DataFrame: Loaded dataset or None if error
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Successfully loaded Obesity dataset")
            print(f"  Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"✗ File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"✗ Error loading file: {str(e)}")
            return None
    
    def get_data_info(self) -> dict:
        """Get information about the dataset"""
        if self.df is None:
            return {}
        
        info = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.to_dict(),
            "target_variable": self.target_column,
            "target_classes": self.df[self.target_column].unique().tolist() if self.target_column in self.df.columns else None,
        }
        return info
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of numeric columns"""
        if self.df is None:
            return pd.DataFrame()
        
        return self.df.describe()
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the dataset by encoding categorical variables
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X features, y target)
        """
        if self.df is None:
            print("✗ Dataset not loaded. Call load_data() first.")
            return None, None
        
        df_processed = self.df.copy()
        
        # Separate features and target
        X = df_processed.drop(columns=[self.target_column])
        y = df_processed[self.target_column]
        
        # Identify categorical and numeric columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        self.label_encoders[self.target_column] = le_target
        
        self.X = X
        self.y = y
        self.feature_columns = list(X.columns)
        
        print(f"✓ Data preprocessing completed")
        print(f"  Features: {len(self.feature_columns)} columns")
        print(f"  Target classes: {len(np.unique(y))} classes")
        
        return X, y
    
    def train_test_split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X is None or self.y is None:
            print("✗ Data not preprocessed. Call preprocess_data() first.")
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"✓ Train-test split completed")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_columns(self) -> list:
        """Get list of feature column names"""
        if self.feature_columns is None:
            if self.df is not None:
                return [col for col in self.df.columns if col != self.target_column]
        return self.feature_columns
    
    def decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Decode encoded predictions back to original class labels
        
        Args:
            predictions (np.ndarray): Encoded predictions
            
        Returns:
            np.ndarray: Original class labels
        """
        if self.target_column not in self.label_encoders:
            print("✗ Target encoder not found")
            return predictions
        
        le_target = self.label_encoders[self.target_column]
        return le_target.inverse_transform(predictions)
    
    def display_value_counts(self) -> dict:
        """Get value counts for the target variable"""
        if self.df is None:
            return {}
        
        return self.df[self.target_column].value_counts().to_dict()


# Example usage
if __name__ == "__main__":
    loader = ObesityDatasetLoader()
    
    # Load data
    df = loader.load_data()
    
    if df is not None:
        # Get information
        info = loader.get_data_info()
        print("\nDataset Info:")
        print(f"Rows: {info['rows']}, Columns: {info['columns']}")
        print(f"Target classes: {info['target_classes']}")
        
        # Get statistics
        print("\nSummary Statistics:")
        print(loader.get_summary_statistics())
        
        # Preprocess
        X, y = loader.preprocess_data()
        
        # Train-test split
        X_train, X_test, y_train, y_test = loader.train_test_split_data(test_size=0.2)
        
        # Show target distribution
        print("\nTarget Variable Distribution:")
        print(loader.display_value_counts())
