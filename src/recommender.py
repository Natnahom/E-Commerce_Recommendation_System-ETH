import pandas as pd
import numpy as np
from model_training import EthiopianRecommenderModel
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class EthiopianEcommerceRecommender:
    def __init__(self, config_path='../config.yaml'):
        """Initialize the recommender system"""
        self.model = EthiopianRecommenderModel(config_path)
        
        # Try to load existing model, otherwise train new one
        try:
            self.model.load_model()
            print("✅ Model loaded successfully")
        except:
            print("⚠️ No saved model found. Training new model...")
            self.model.load_data_from_csv('../data/raw/ethiopian_products_10k.csv')
            self.model.prepare_features()
            self.model.train()
            self.model.save_model()
        
        self.config_path = config_path
        self.user_history = defaultdict(list)  # Simple in-memory user history
        
    def recommend_by_product(self, product_id, user_context=None):
        """Main recommendation method - Content-based filtering"""
        if user_context is None:
            user_context = {}
        
        top_n = user_context.get('top_n', 5)
        diversify = user_context.get('diversify', True)
        
        # Check if product exists
        if product_id not in self.model.df['product_id'].values:
            print(f"⚠️ Product {product_id} not found. Using cold start strategy.")
            return self._cold_start_recommendations(user_context)
        
        # Get content-based recommendations
        weights = user_context.get('weights', {
            'content': 0.6,    # TF-IDF similarity (Content-based)
            'category': 0.3,   # Category similarity
            'popularity': 0.1  # Popularity score
        })
        
        recommendations = self.model.get_recommendations(
            product_id=product_id,
            top_n=top_n,
            weights=weights,
            diversify=diversify
        )
        
        # Apply user filters
        filtered_recs = self._apply_filters(recommendations, user_context)
        
        # Store in user history for session-based recommendations
        user_id = user_context.get('user_id')
        if user_id:
            self.user_history[user_id].append(product_id)
        
        return {
            'query_product': self.get_product_details(product_id),
            'recommendations': filtered_recs,
            'context': user_context,
            'strategy': 'content_based_filtering',
            'model_type': 'unsupervised_content_based'
        }
    
    def recommend_by_text_search(self, search_text, user_context=None):
        """Recommend based on text search using content similarity"""
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
                'delivery_available': product.get('delivery_available', 'Unknown'),
                'location': product.get('location', 'Unknown')
            })
        
        return {
            'search_query': search_text,
            'recommendations': recommendations[:top_n],
            'context': user_context,
            'strategy': 'text_search_based'
        }
    
    def recommend_by_user_history(self, user_id, user_context=None):
        """Recommend based on user's browsing history (Session-based)"""
        if user_context is None:
            user_context = {}
        
        top_n = user_context.get('top_n', 5)
        
        # Get user's browsing history
        history = self.user_history.get(user_id, [])
        if not history:
            return self._get_popular_recommendations(user_context)
        
        # Aggregate recommendations from history
        all_recommendations = []
        for product_id in history[-3:]:  # Last 3 viewed products
            recs = self.model.get_recommendations(product_id, top_n=top_n*2)
            all_recommendations.extend(recs)
        
        # Remove duplicates and products in history
        seen = set(history)
        unique_recs = []
        seen_recs = set()
        
        for rec in all_recommendations:
            rec_id = rec['product_id']
            if rec_id not in seen and rec_id not in seen_recs:
                unique_recs.append(rec)
                seen_recs.add(rec_id)
        
        # Sort by similarity score and take top N
        unique_recs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'user_id': user_id,
            'recommendations': unique_recs[:top_n],
            'context': user_context,
            'strategy': 'user_history_based',
            'history_size': len(history)
        }
    
    def recommend_by_category(self, category, user_context=None):
        """Recommend popular items in a category"""
        if user_context is None:
            user_context = {}
        
        top_n = user_context.get('top_n', 5)
        
        # Filter products by category
        category_products = self.model.df[self.model.df['category'] == category]
        
        if len(category_products) == 0:
            return {"error": f"Category '{category}' not found"}
        
        # Sort by popularity score
        category_products = category_products.copy()
        category_products['popularity'] = self.model.popularity_scores[category_products.index]
        category_products = category_products.sort_values('popularity', ascending=False)
        
        recommendations = []
        for _, product in category_products.head(top_n).iterrows():
            recommendations.append({
                'product_id': product['product_id'],
                'name': product['name'],
                'category': product['category'],
                'subcategory': product.get('subcategory', ''),
                'price': float(product['price']),
                'rating': float(product['rating']),
                'popularity_score': float(product['popularity']),
                'delivery_available': product.get('delivery_available', 'Unknown'),
                'location': product.get('location', 'Unknown')
            })
        
        return {
            'category': category,
            'recommendations': recommendations,
            'context': user_context,
            'strategy': 'category_popularity'
        }
    
    def _cold_start_recommendations(self, user_context):
        """Handle cold start scenarios with popular items"""
        top_n = user_context.get('top_n', 5)
        
        # Use popular items with diversification
        recs = self.model._get_popular_recommendations(top_n * 2, diversify=True)
        
        # Apply filters
        filtered_recs = self._apply_filters(recs, user_context)
        
        return {
            'query_product': None,
            'recommendations': filtered_recs[:top_n],
            'context': user_context,
            'strategy': 'cold_start_popular',
            'explanation': 'Product not found, showing popular items instead'
        }
    
    def _get_popular_recommendations(self, user_context):
        """Get popular recommendations"""
        top_n = user_context.get('top_n', 5)
        diversify = user_context.get('diversify', True)
        
        recs = self.model._get_popular_recommendations(top_n * 2, diversify)
        filtered_recs = self._apply_filters(recs, user_context)
        
        return {
            'recommendations': filtered_recs[:top_n],
            'context': user_context,
            'strategy': 'popular_items'
        }
    
    def _apply_filters(self, recommendations, user_context):
        """Apply user-specific filters to recommendations"""
        if not recommendations:
            return []
        
        filtered = []
        
        min_price = user_context.get('min_price')
        max_price = user_context.get('max_price')
        user_location = user_context.get('location')
        filter_delivery = user_context.get('filter_delivery', False)
        required_category = user_context.get('category')
        
        for rec in recommendations:
            # Price filters
            if min_price is not None and rec['price'] < min_price:
                continue
            if max_price is not None and rec['price'] > max_price:
                continue
            
            # Location and delivery filters
            if filter_delivery and user_location:
                if rec.get('location') != user_location and rec.get('delivery_available') == 'No':
                    continue
            
            # Category filter
            if required_category and rec['category'] != required_category:
                continue
            
            filtered.append(rec)
        
        return filtered
    
    def get_product_details(self, product_id):
        """Get details of a specific product"""
        product = self.model.df[self.model.df['product_id'] == product_id]
        
        if len(product) == 0:
            return None
        
        product = product.iloc[0]
        
        details = {
            'product_id': product['product_id'],
            'name': product['name'],
            'description': product['description'],
            'category': product['category'],
            'subcategory': product.get('subcategory', ''),
            'price': float(product['price']),
            'rating': float(product['rating']),
            'brand': product.get('brand', 'Unknown'),
            'location': product.get('location', 'Unknown'),
            'delivery_available': product.get('delivery_available', 'Unknown'),
        }
        
        # Add optional fields if they exist
        optional_fields = ['description_amharic', 'stock_status', 'stock_quantity', 'tags']
        for field in optional_fields:
            if field in product:
                details[field] = product[field]
        
        return details
    
    def get_product_categories(self):
        """Get all available product categories"""
        return sorted(self.model.df['category'].unique().tolist())
    
    def get_stats(self):
        """Get system statistics"""
        df = self.model.df
        
        return {
            'total_products': len(df),
            'categories': len(df['category'].unique()),
            'average_price': float(df['price'].mean()),
            'average_rating': float(df['rating'].mean()),
            'locations': sorted(df['location'].unique().tolist()) if 'location' in df.columns else [],
            'model_type': 'content_based_unsupervised',
            'algorithm': 'TF-IDF + Cosine Similarity'
        }
    
    def explain_recommendation(self, query_product_id, recommended_product_id):
        """Explain why a product was recommended (Content-based reasoning)"""
        query_product = self.get_product_details(query_product_id)
        recommended_product = self.get_product_details(recommended_product_id)
        
        if not query_product or not recommended_product:
            return {"error": "One or both products not found"}
        
        # Find common features
        common_features = []
        
        if query_product['category'] == recommended_product['category']:
            common_features.append(f"Same category: {query_product['category']}")
        
        if query_product.get('brand') == recommended_product.get('brand'):
            common_features.append(f"Same brand: {query_product.get('brand')}")
        
        # Check price similarity
        price_ratio = min(query_product['price'], recommended_product['price']) / max(query_product['price'], recommended_product['price'])
        if price_ratio > 0.7:
            common_features.append(f"Similar price range ({price_ratio:.0%} difference)")
        
        return {
            'query_product': query_product['name'],
            'recommended_product': recommended_product['name'],
            'common_features': common_features,
            'explanation': f"Recommended based on content similarity: {', '.join(common_features) if common_features else 'General popularity'}"
        }


# Main execution for recommender.py
if __name__ == "__main__":
    print("="*60)
    print("🛍️ ETHIOPIAN E-COMMERCE RECOMMENDER SYSTEM")
    print("="*60)
    
    # Initialize the recommender
    recommender = EthiopianEcommerceRecommender()
    
    # Get system statistics
    print("\n📊 SYSTEM STATISTICS:")
    print("-"*40)
    stats = recommender.get_stats()
    print(f"• Total Products: {stats['total_products']:,}")
    print(f"• Categories: {stats['categories']}")
    print(f"• Average Price: {stats['average_price']:,.0f} ETB")
    print(f"• Average Rating: {stats['average_rating']:.1f}/5.0")
    print(f"• Model Type: {stats['model_type']}")
    print(f"• Algorithm: {stats['algorithm']}")
    
    # Get available categories
    print("\n📂 AVAILABLE CATEGORIES:")
    print("-"*40)
    categories = recommender.get_product_categories()
    for i, category in enumerate(categories[:10], 1):
        print(f"{i}. {category}")
    if len(categories) > 10:
        print(f"... and {len(categories) - 10} more")
    
    # Test recommendations
    print("\n🧪 TESTING RECOMMENDATIONS:")
    print("-"*40)
    
    if stats['total_products'] > 0:
        # Get first product for testing
        test_product_id = recommender.model.df.iloc[0]['product_id']
        test_product_name = recommender.model.df.iloc[0]['name']
        
        print(f"\n1. Content-based recommendations for:")
        print(f"   Product: {test_product_name}")
        print(f"   ID: {test_product_id}")
        
        # Get content-based recommendations
        recommendations = recommender.recommend_by_product(
            test_product_id,
            user_context={
                'top_n': 3,
                'diversify': True
            }
        )
        
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"\n   {i}. {rec['name']}")
            print(f"      Category: {rec['category']}")
            print(f"      Price: {rec['price']:,.0f} ETB")
            print(f"      Rating: {rec['rating']:.1f}/5.0")
            print(f"      Similarity: {rec['similarity_score']:.3f}")
        
        # Test category-based recommendations
        if categories:
            print(f"\n2. Category-based recommendations (Fashion):")
            category_recs = recommender.recommend_by_category('Fashion', {'top_n': 2})
            if 'recommendations' in category_recs:
                for i, rec in enumerate(category_recs['recommendations'], 1):
                    print(f"   {i}. {rec['name']}")
                    print(f"      Price: {rec['price']:,.0f} ETB | Popularity: {rec['popularity_score']:.3f}")
        
        # Test text search
        print(f"\n3. Text search recommendations (search: 'coffee'):")
        search_recs = recommender.recommend_by_text_search('coffee', {'top_n': 2})
        if 'recommendations' in search_recs:
            for i, rec in enumerate(search_recs['recommendations'], 1):
                print(f"   {i}. {rec['name']}")
                print(f"      Category: {rec['category']}")
                print(f"      Similarity: {rec['similarity_score']:.3f}")
    
    print("\n" + "="*60)
    print("✅ RECOMMENDER SYSTEM READY")
    print("="*60)