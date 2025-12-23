import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from data_preprocessing import EthiopianProductPreprocessor

class EthiopianRecommenderModel:
    def __init__(self, config_path='config.yaml'):
        self.preprocessor = EthiopianProductPreprocessor(config_path)
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
    def train(self):
        """Train the TF-IDF model and compute similarity matrix"""
        # Load and preprocess data
        self.df = self.preprocessor.load_processed_data()
        
        # Initialize TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform
        print("Training TF-IDF model...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        
        # Compute cosine similarity
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"Model trained on {len(self.df)} products")
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        return self
    
    def save_model(self):
        """Save trained model and data"""
        model_data = {
            'tfidf': self.tfidf,
            'similarity_matrix': self.similarity_matrix,
            'df': self.df
        }
        
        with open('../data/models/tfidf_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("Model saved successfully")
    
    def load_model(self):
        """Load trained model"""
        with open('../data/models/tfidf_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf = model_data['tfidf']
        self.similarity_matrix = model_data['similarity_matrix']
        self.df = model_data['df']
        print("Model loaded successfully")
        return self
    
    def get_recommendations(self, product_id, top_n=5, user_location=None, 
                           filter_delivery=True, min_price=None, max_price=None):
        """Get recommendations for a product"""
        if self.df is None or self.similarity_matrix is None:
            self.load_model()
        
        # Find product index
        if product_id in self.df['product_id'].values:
            idx = self.df[self.df['product_id'] == product_id].index[0]
        else:
            # If product not found, use first product
            idx = 0
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (excluding the product itself)
        recommendations = []
        for i, score in similarity_scores[1:top_n+1]:
            product = self.df.iloc[i]
            
            # Apply filters
            if filter_delivery and user_location:
                if product['location'] != user_location and product['delivery_available'] == 'No':
                    continue
            
            if min_price and product['price'] < min_price:
                continue
            
            if max_price and product['price'] > max_price:
                continue
            
            recommendations.append({
                'product_id': product['product_id'],
                'name': product['name'],
                'description': product['description'],
                'category': product['category'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'similarity_score': float(score),
                'delivery_available': product['delivery_available'],
                'location': product['location']
            })
        
        return recommendations

if __name__ == "__main__":
    # Train and save model
    model = EthiopianRecommenderModel()
    model.train()
    model.save_model()
    
    # Test recommendations
    print("\nTesting recommendations for product P001:")
    recommendations = model.get_recommendations("P001", top_n=3)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} - Similarity: {rec['similarity_score']:.3f}")