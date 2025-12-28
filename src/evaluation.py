import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity as pairwise_cosine

class RecommenderEvaluator:
    """Evaluation metrics for the content-based recommender"""
    
    def __init__(self, recommender):
        self.recommender = recommender
        self.df = recommender.model.df
    
    def calculate_intra_cluster_similarity(self, category, sample_size=10000):
        """Calculate similarity within a category (cohesion)"""
        category_products = self.df[self.df['category'] == category]

        if len(category_products) < 2:
            return 0

        # 🔹 Sample up to sample_size products from this category
        if len(category_products) > sample_size:
            category_products = category_products.sample(sample_size, random_state=42)

        indices = category_products.index.tolist()
        vectors = self.recommender.model.tfidf_matrix[indices]

        # Vectorized cosine similarity
        sims = pairwise_cosine(vectors)
        intra_sim = np.mean(sims[np.triu_indices_from(sims, k=1)])
        return intra_sim

    def calculate_inter_cluster_dissimilarity(self, category1, category2, sample_size=10000):
        """Calculate dissimilarity between categories (separation)"""
        cat1_products = self.df[self.df['category'] == category1]
        cat2_products = self.df[self.df['category'] == category2]

        if len(cat1_products) == 0 or len(cat2_products) == 0:
            return 0

        # 🔹 Sample each category
        if len(cat1_products) > sample_size:
            cat1_products = cat1_products.sample(sample_size, random_state=42)
        if len(cat2_products) > sample_size:
            cat2_products = cat2_products.sample(sample_size, random_state=42)

        idx1 = cat1_products.index.tolist()
        idx2 = cat2_products.index.tolist()
        vectors1 = self.recommender.model.tfidf_matrix[idx1]
        vectors2 = self.recommender.model.tfidf_matrix[idx2]

        sims = pairwise_cosine(vectors1, vectors2)
        return np.mean(sims)
        
    def calculate_coverage(self, top_n=5):
        """Calculate what percentage of products get recommended"""
        all_products = set(self.df['product_id'])
        recommended_products = set()
        
        for product_id in self.df['product_id'].head(20):  # Sample for efficiency
            try:
                recommendations = self.recommender.recommend_by_product(
                    product_id, 
                    {'top_n': top_n}
                )
                for rec in recommendations['recommendations']:
                    recommended_products.add(rec['product_id'])
            except:
                continue
        
        coverage = len(recommended_products) / len(all_products)
        return coverage
    
    def generate_similarity_report(self):
        """Generate comprehensive similarity analysis"""
        categories = self.df['category'].unique()
        report = {
            'categories': [],
            'intra_similarity': [],
            'sample_products': []
        }
        
        for category in categories[:5]:  # Limit to first 5 for speed
            intra_sim = self.calculate_intra_cluster_similarity(category)
            
            report['categories'].append(category)
            report['intra_similarity'].append(intra_sim)
            
            # Get sample products from category
            sample = self.df[self.df['category'] == category].head(2)
            report['sample_products'].append(
                sample[['product_id', 'name']].to_dict('records')
            )
        
        return pd.DataFrame(report)
    
    def plot_similarity_heatmap(self, save_path='similarity_heatmap.png'):
        """Plot similarity heatmap for top categories"""
        top_categories = self.df['category'].value_counts().head(6).index.tolist()
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(top_categories), len(top_categories)))
        
        for i, cat1 in enumerate(top_categories):
            for j, cat2 in enumerate(top_categories):
                if i == j:
                    similarity_matrix[i][j] = self.calculate_intra_cluster_similarity(cat1)
                else:
                    similarity_matrix[i][j] = self.calculate_inter_cluster_dissimilarity(cat1, cat2)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=top_categories,
            yticklabels=top_categories,
            cmap='YlOrRd'
        )
        plt.title('Category Similarity Matrix')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        return similarity_matrix

def row_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two sparse vectors"""
    from numpy.linalg import norm
    v1 = vec1.toarray().flatten()
    v2 = vec2.toarray().flatten()
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10))

# Example usage
if __name__ == "__main__":
    from recommender import EthiopianEcommerceRecommender
    
    recommender = EthiopianEcommerceRecommender()
    evaluator = RecommenderEvaluator(recommender)
    
    print("Evaluation Report:")
    print("=" * 50)
    
    # Calculate coverage
    coverage = evaluator.calculate_coverage()
    print(f"Catalog Coverage: {coverage:.2%}")
    
    # Generate similarity report
    report = evaluator.generate_similarity_report()
    print("\nCategory Similarity Analysis:")
    print(report.to_string(index=False))
    
    # Plot heatmap
    print("\nGenerating similarity heatmap...")
    evaluator.plot_similarity_heatmap()