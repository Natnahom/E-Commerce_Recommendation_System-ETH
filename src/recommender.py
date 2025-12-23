import pandas as pd
import numpy as np
from model_training import EthiopianRecommenderModel
import json
from sklearn.metrics.pairwise import cosine_similarity

class EthiopianEcommerceRecommender:
    def __init__(self, config_path='config.yaml'):
        self.model = EthiopianRecommenderModel(config_path)
        self.model.load_model()
        self.config_path = config_path
        
    def recommend_by_product(self, product_id, user_context=None):
        """Main recommendation method"""
        if user_context is None:
            user_context = {}
        
        top_n = user_context.get('top_n', 5)
        user_location = user_context.get('location', 'Addis Ababa')
        filter_delivery = user_context.get('filter_delivery', True)
        min_price = user_context.get('min_price')
        max_price = user_context.get('max_price')
        
        recommendations = self.model.get_recommendations(
            product_id=product_id,
            top_n=top_n,
            user_location=user_location,
            filter_delivery=filter_delivery,
            min_price=min_price,
            max_price=max_price
        )
        
        return {
            'query_product': self.get_product_details(product_id),
            'recommendations': recommendations,
            'context': user_context
        }
    
    def recommend_by_text_search(self, search_text, user_context=None):
        """Recommend based on text search"""
        if user_context is None:
            user_context = {}
        
        # Transform search text using TF-IDF
        search_vector = self.model.tfidf.transform([search_text])
        
        # Compute similarity with all products
        similarities = cosine_similarity(search_vector, self.model.tfidf_matrix)
        
        # Get top similar products
        similar_indices = similarities.argsort()[0][::-1]
        
        recommendations = []
        top_n = user_context.get('top_n', 5)
        
        for idx in similar_indices[:top_n]:
            product = self.model.df.iloc[idx]
            recommendations.append({
                'product_id': product['product_id'],
                'name': product['name'],
                'description': product['description'],
                'category': product['category'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'similarity_score': float(similarities[0][idx]),
                'delivery_available': product['delivery_available'],
                'location': product['location']
            })
        
        return {
            'search_query': search_text,
            'recommendations': recommendations,
            'context': user_context
        }
    
    def get_product_details(self, product_id):
        """Get details of a specific product"""
        product = self.model.df[self.model.df['product_id'] == product_id]
        
        if len(product) == 0:
            return None
        
        product = product.iloc[0]
        return {
            'product_id': product['product_id'],
            'name': product['name'],
            'description': product['description'],
            'description_amharic': product.get('description_amharic', ''),
            'category': product['category'],
            'subcategory': product.get('subcategory', ''),
            'price': float(product['price']),
            'rating': float(product['rating']),
            'brand': product.get('brand', ''),
            'location': product['location'],
            'delivery_available': product['delivery_available'],
            'stock_status': product.get('stock_status', 'Unknown')
        }
    
    def get_product_categories(self):
        """Get all available product categories"""
        return sorted(self.model.df['category'].unique().tolist())
    
    def get_stats(self):
        """Get system statistics"""
        return {
            'total_products': len(self.model.df),
            'categories': len(self.model.df['category'].unique()),
            'average_price': float(self.model.df['price'].mean()),
            'average_rating': float(self.model.df['rating'].mean()),
            'locations': sorted(self.model.df['location'].unique().tolist())
        }

# Example usage
if __name__ == "__main__":
    recommender = EthiopianEcommerceRecommender()
    
    print("System Statistics:")
    print(json.dumps(recommender.get_stats(), indent=2))
    
    print("\nCategories available:")
    print(recommender.get_product_categories())
    
    print("\nRecommendations for smartphone (P001):")
    recommendations = recommender.recommend_by_product("P001", {
        'location': 'Addis Ababa',
        'filter_delivery': True
    })
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"{i}. {rec['name']} - ${rec['price']} (Similarity: {rec['similarity_score']:.3f})")