import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

class EthiopianProductPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self):
        """Load Ethiopian product data from CSV"""
        df = pd.read_csv(self.config['data']['raw_path'])
        print(f"Loaded {len(df)} products from dataset")
        return df
    
    def clean_text(self, text):
        """Clean text data for Amharic and English"""
        if pd.isna(text):
            return ""
        # Remove extra whitespace
        text = str(text).strip()
        # Remove special characters but keep Amharic and English letters
        text = re.sub(r'[^\w\s\u1200-\u137F]', ' ', text)
        return text.lower()
    
    def preprocess_data(self, df):
        """Main preprocessing pipeline"""
        # Clean text columns
        text_cols = self.config['features']['text_columns']
        for col in text_cols:
            if col in df.columns:
                df[f'{col}_cleaned'] = df[col].apply(self.clean_text)
        
        # Handle missing values
        df['rating'] = df['rating'].fillna(df['rating'].median())
        df['price'] = df['price'].fillna(df['price'].median())
        
        # Filter by price and rating
        df = df[
            (df['price'] >= self.config['preprocessing']['min_price']) &
            (df['price'] <= self.config['preprocessing']['max_price']) &
            (df['rating'] >= self.config['preprocessing']['min_rating']) &
            (df['rating'] <= self.config['preprocessing']['max_rating'])
        ]
        
        # Create combined text feature for TF-IDF
        df['combined_features'] = df.apply(
            lambda row: ' '.join([
                str(row.get(f'{col}_cleaned', '')) 
                for col in text_cols 
                if col in df.columns
            ]), axis=1
        )
        
        print(f"Preprocessed {len(df)} products")
        return df
    
    def save_processed_data(self, df, path=None):
        """Save processed data"""
        if path is None:
            path = self.config['data']['processed_path']
        
        with open(path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Saved processed data to {path}")
    
    def load_processed_data(self, path=None):
        """Load processed data"""
        if path is None:
            path = self.config['data']['processed_path']
        
        with open(path, 'rb') as f:
            df = pickle.load(f)
        print(f"Loaded processed data from {path}")
        return df

if __name__ == "__main__":
    preprocessor = EthiopianProductPreprocessor()
    df = preprocessor.load_data()
    df_processed = preprocessor.preprocess_data(df)
    preprocessor.save_processed_data(df_processed)