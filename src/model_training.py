import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class EthiopianRecommenderModel:
    def __init__(self, config_path='../config.yaml'):
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.category_matrix = None
        self.price_buckets = None
        self.popularity_scores = None
        
    def load_data_from_csv(self, csv_path='../data/raw/ethiopian_products_10k.csv'):
        """Load data directly from CSV file"""
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} products")
        return self
    
    def prepare_features(self):
        if self.df is None:
            print("Error: No data loaded. Call load_data_from_csv() first.")
            return self

        print("Preparing features...")
        for col in ['name', 'description', 'description_amharic', 'brand', 'category', 'subcategory', 'tags']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')

        self.df['comprehensive_features'] = self.df.apply(
            lambda x: " ".join([
                str(x.get('name', '')),
                str(x.get('description', '')),
                str(x.get('description_amharic', '')),
                str(x.get('brand', '')),
                str(x.get('category', '')),
                str(x.get('subcategory', '')),
                str(x.get('tags', ''))
            ]),
            axis=1
        )
        return self
    
    def train(self):
        """Train the content-based recommendation model using unsupervised learning"""
        if self.df is None:
            print("Error: No data loaded. Call load_data_from_csv() first.")
            return self
        
        print(f"Training model on {len(self.df)} products...")
        
        # 1. Content-based filtering using TF-IDF (Unsupervised)
        print("Training content-based model with TF-IDF...")
        self._train_content_based()
        
        # 2. Train category-based similarity (Unsupervised)
        print("Training category-based model...")
        self._train_category_based()
        
        # 3. Train price-based similarity (Unsupervised)
        print("Training price-based model...")
        self._train_price_based()
        
        # 4. Calculate popularity scores (Unsupervised)
        print("Calculating popularity scores...")
        self._calculate_popularity_scores()
        
        print("✅ Model training complete!")
        return self
    
    def _train_content_based(self):
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['comprehensive_features'])
        self.similarity_matrix = None
        print("Using on-demand content similarity (no full matrix).")

    def _train_category_based(self):
        # We'll compute category score on-demand: 1.0 if same category, else 0.5
        self.category_matrix = None
        print("Using rule-based category scores (no matrix).")        
        
    def _train_price_based(self):
        """Create price buckets for price-based recommendations (Unsupervised)"""
        # Create price buckets using quantiles
        prices = self.df['price'].values
        self.price_buckets = pd.qcut(prices, q=10, labels=False, duplicates='drop')
        
    def _calculate_popularity_scores(self):
        """Calculate popularity scores based on product features (Unsupervised)"""
        # Normalize rating to 0-1 scale
        rating_norm = (self.df['rating'] - 1) / 4  # 1-5 scale to 0-1
        
        # Calculate stock score
        if 'stock_quantity' in self.df.columns:
            stock_score = self.df['stock_quantity'].apply(
                lambda x: 1 if x > 10 else (0.5 if x > 0 else 0.1)
            )
        else:
            stock_score = pd.Series([0.5] * len(self.df))
        
        # Calculate delivery score
        if 'delivery_available' in self.df.columns:
            delivery_score = self.df['delivery_available'].apply(
                lambda x: 1 if x == 'Yes' else 0.5
            )
        else:
            delivery_score = pd.Series([0.5] * len(self.df))
        
        # Combine scores (Unsupervised weighting)
        self.popularity_scores = (
            0.5 * rating_norm + 
            0.3 * stock_score + 
            0.2 * delivery_score
        ).values
        
    def get_recommendations(self, product_id, top_n=10, weights=None, diversify=True):
        """Get content-based recommendations using unsupervised learning with diversity and deduplication"""
        if self.df is None or self.tfidf_matrix is None:
            print("Error: Model not trained. Call train() first.")
            return []

        if product_id not in self.df['product_id'].values:
            print(f"Product {product_id} not found. Using popular items as fallback.")
            return self._get_popular_recommendations(top_n, diversify)

        # Get product index
        idx = self.df.index[self.df['product_id'] == product_id][0]

        # Default weights
        if weights is None:
            weights = {'content': 0.5, 'category': 0.3, 'popularity': 0.2}

        # --- Content similarity on demand ---
        query_vec = self.tfidf_matrix[idx]
        content_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # --- Category score (rule-based, no matrix) ---
        query_cat = self.df.iloc[idx]['category']
        query_sub = self.df.iloc[idx].get('subcategory', None)
        cats = self.df['category'].values
        subs = self.df['subcategory'].values if 'subcategory' in self.df.columns else np.array([None]*len(self.df))
        category_scores = np.where(cats == query_cat, 1.0, 0.5)
        if query_sub is not None:
            category_scores = np.where((cats == query_cat) & (subs == query_sub), 1.05, category_scores)

        # --- Price scores ---
        price_bucket = self.price_buckets[idx]
        price_scores = np.where(self.price_buckets == price_bucket, 1.0, 0.5)

        # --- Combine scores ---
        combined_scores = (
            weights['content'] * content_scores +
            weights['category'] * category_scores +
            0.1 * price_scores +
            weights['popularity'] * self.popularity_scores
        )

        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]

        recommendations = []
        seen_categories = set()
        seen_names = set()

        for i in sorted_indices:
            if i == idx:
                continue
            product = self.df.iloc[i]
            category = product['category']

            # Deduplicate by product name
            if product['name'] in seen_names:
                continue
            seen_names.add(product['name'])

            # Diversification: enforce category variety after first few
            if diversify and len(recommendations) >= 3:
                if category in seen_categories and len(seen_categories) > 1:
                    continue
            seen_categories.add(category)

            rec = {
                'product_id': product['product_id'],
                'name': product['name'],
                'category': category,
                'subcategory': product.get('subcategory', ''),
                'price': float(product['price']),
                'rating': float(product['rating']),
                'similarity_score': float(combined_scores[i]),
                'content_score': float(content_scores[i]),
                'category_score': float(category_scores[i]),
                'popularity_score': float(self.popularity_scores[i]),
            }
            for field in ['delivery_available', 'location', 'stock_status', 'brand']:
                if field in product:
                    rec[field] = product[field]

            recommendations.append(rec)
            if len(recommendations) >= top_n:
                break

        return recommendations
    
    def _get_popular_recommendations(self, top_n=10, diversify=True):
        """Get popular recommendations as fallback (Unsupervised)"""
        # Sort by popularity score
        popular_indices = np.argsort(self.popularity_scores)[::-1]
        
        recommendations = []
        seen_categories = set()
        
        for idx in popular_indices[:top_n * 3]:  # Look at more for diversity
            product = self.df.iloc[idx]
            category = product['category']
            
            # Apply diversification
            if diversify and len(recommendations) >= 3:
                if category in seen_categories and len(seen_categories) > 1:
                    continue
            
            seen_categories.add(category)
            
            rec = {
                'product_id': product['product_id'],
                'name': product['name'],
                'category': category,
                'subcategory': product.get('subcategory', ''),
                'price': float(product['price']),
                'rating': float(product['rating']),
                'popularity_score': float(self.popularity_scores[idx]),
            }
            
            # Add optional fields
            for field in ['delivery_available', 'location', 'stock_status', 'brand']:
                if field in product:
                    rec[field] = product[field]
            
            recommendations.append(rec)
            
            if len(recommendations) >= top_n:
                break
        
        return recommendations
    
    def get_cold_start_recommendations(self, product_features=None, top_n=10):
        """Get recommendations for cold start products (Unsupervised)"""
        if product_features is None:
            return self._get_popular_recommendations(top_n, diversify=True)
        
        # Create feature vector for the new product
        feature_text = f"{product_features.get('name', '')} " \
                      f"{product_features.get('description', '')} " \
                      f"{product_features.get('category', '')} " \
                      f"{product_features.get('brand', '')}"
        
        # Transform using trained TF-IDF
        new_vector = self.tfidf.transform([feature_text])
        
        # Compute similarity with existing products
        similarities = cosine_similarity(new_vector, self.tfidf_matrix)[0]
        
        # Combine with popularity for cold start
        cold_start_scores = 0.6 * similarities + 0.4 * self.popularity_scores
        
        # Get recommendations
        sorted_indices = np.argsort(cold_start_scores)[::-1]
        
        recommendations = []
        for idx in sorted_indices[:top_n]:
            product = self.df.iloc[idx]
            recommendations.append({
                'product_id': product['product_id'],
                'name': product['name'],
                'category': product['category'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'cold_start_score': float(cold_start_scores[idx]),
                'delivery_available': product.get('delivery_available', 'Unknown')
            })
        
        return recommendations
    
    def save_model(self, filepath='../data/models/content_based_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'tfidf': self.tfidf,
            'tfidf_matrix': self.tfidf_matrix,
            'price_buckets': self.price_buckets,
            'popularity_scores': self.popularity_scores,
            'df': self.df
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved successfully to {filepath}")

    def load_model(self, filepath='../data/models/content_based_model.pkl'):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)  # FIX
            self.tfidf = model_data['tfidf']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.price_buckets = model_data['price_buckets']
            self.popularity_scores = model_data['popularity_scores']
            self.df = model_data['df']
            self.similarity_matrix = None
            self.category_matrix = None
            print("✅ Model loaded successfully")
        except FileNotFoundError:
            print(f"❌ Model file not found at {filepath}")
            print("   Train a new model or check the path.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        return self
    
    def evaluate_model(self, sample_size=100, random_state=42):
        """Evaluate the content-based recommendation model"""
        print("\n" + "="*60)
        print("📊 CONTENT-BASED MODEL EVALUATION")
        print("="*60)
        
        if self.df is None:
            print("Error: No data loaded.")
            return {}
        
        # Sample products for evaluation
        test_products = self.df.sample(min(sample_size, len(self.df)), 
                                      random_state=random_state)
        
        coverage_set = set()
        category_match_rates = []
        price_compatibility_rates = []
        
        print(f"Testing on {len(test_products)} products...")
        
        for _, product in test_products.iterrows():
            try:
                recs = self.get_recommendations(
                    product['product_id'],
                    top_n=5,
                    diversify=True
                )
                
                if recs:
                    # Track coverage
                    for rec in recs:
                        coverage_set.add(rec['product_id'])
                    
                    # Calculate category match rate
                    query_category = product['category']
                    category_matches = sum(1 for rec in recs if rec['category'] == query_category)
                    category_match_rates.append(category_matches / len(recs))
                    
                    # Calculate price compatibility
                    query_price = product['price']
                    price_compatible = sum(1 for rec in recs if 
                                         abs(rec['price'] - query_price) / query_price <= 0.5)
                    price_compatibility_rates.append(price_compatible / len(recs))
                    
            except Exception as e:
                continue
        
        # Calculate metrics
        coverage = len(coverage_set) / len(self.df) * 100 if len(self.df) > 0 else 0
        avg_category_match = np.mean(category_match_rates) * 100 if category_match_rates else 0
        avg_price_compatibility = np.mean(price_compatibility_rates) * 100 if price_compatibility_rates else 0
        
        print("\n📈 EVALUATION RESULTS:")
        print("-"*40)
        print(f"• Catalog Coverage: {coverage:.1f}% of products recommended")
        print(f"• Category Match Rate: {avg_category_match:.1f}%")
        print(f"• Price Compatibility: {avg_price_compatibility:.1f}%")
        print(f"• Unique Recommended: {len(coverage_set)} products")
        
        # Overall score
        overall_score = np.mean([
            min(coverage, 100),
            avg_category_match,
            avg_price_compatibility
        ])
        
        print(f"\n📊 OVERALL SCORE: {overall_score:.1f}/100")
        
        if overall_score >= 70:
            print("✅ EXCELLENT: Model is performing very well!")
        elif overall_score >= 50:
            print("⚠️ GOOD: Model is performing adequately.")
        else:
            print("❌ NEEDS IMPROVEMENT: Model performance could be better.")
        
        return {
            'coverage': coverage,
            'category_match': avg_category_match,
            'price_compatibility': avg_price_compatibility,
            'overall_score': overall_score
        }


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🎯 ETHIOPIAN E-COMMERCE RECOMMENDER SYSTEM")
    print("🎯 Content-Based Filtering with Unsupervised Learning")
    print("="*60)
    
    # Create and train the model
    model = EthiopianRecommenderModel()
    
    # Load data from CSV
    model.load_data_from_csv('ethiopian_products_10k.csv')
    
    # Prepare features
    model.prepare_features()
    
    # Train the model
    model.train()
    
    # Save the model
    model.save_model()
    
    # Evaluate the model
    metrics = model.evaluate_model(sample_size=50)
    
    # Test with a sample product
    print("\n🧪 TEST RECOMMENDATIONS:")
    print("-"*40)
    
    if len(model.df) > 0:
        test_product = model.df.iloc[0]['product_id']
        test_name = model.df.iloc[0]['name']
        
        print(f"Getting recommendations for: {test_name}")
        print(f"Product ID: {test_product}")
        
        recommendations = model.get_recommendations(test_product, top_n=3)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   Category: {rec['category']}")
            print(f"   Price: {rec['price']:,.0f} ETB")
            print(f"   Rating: {rec['rating']:.1f}/5.0")
            print(f"   Similarity Score: {rec['similarity_score']:.3f}")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE")
    print("="*60)