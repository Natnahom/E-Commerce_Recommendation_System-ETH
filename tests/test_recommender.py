import unittest
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from recommender import EthiopianEcommerceRecommender
from data_preprocessing import EthiopianProductPreprocessor

class TestEthiopianRecommender(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.recommender = EthiopianEcommerceRecommender()
        
    def test_system_initialization(self):
        """Test if system initializes correctly"""
        stats = self.recommender.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_products', stats)
        self.assertGreater(stats['total_products'], 0)
    
    def test_product_recommendation(self):
        """Test product-based recommendations"""
        result = self.recommender.recommend_by_product(
            "P001", 
            {'location': 'Addis Ababa', 'top_n': 3}
        )
        
        self.assertIn('query_product', result)
        self.assertIn('recommendations', result)
        self.assertEqual(len(result['recommendations']), 3)
        
        # Check recommendation structure
        recommendation = result['recommendations'][0]
        required_keys = ['product_id', 'name', 'price', 'similarity_score']
        for key in required_keys:
            self.assertIn(key, recommendation)
    
    def test_text_search_recommendation(self):
        """Test text search-based recommendations"""
        result = self.recommender.recommend_by_text_search(
            "smartphone",
            {'top_n': 2}
        )
        
        self.assertIn('search_query', result)
        self.assertIn('recommendations', result)
        self.assertLessEqual(len(result['recommendations']), 2)
    
    def test_product_details(self):
        """Test getting product details"""
        product = self.recommender.get_product_details("P001")
        
        self.assertIsNotNone(product)
        self.assertEqual(product['product_id'], "P001")
        self.assertIn('name', product)
        self.assertIn('price', product)
    
    def test_categories_list(self):
        """Test getting categories"""
        categories = self.recommender.get_product_categories()
        
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)
        self.assertIn('Electronics', categories)
    
    def test_delivery_filter(self):
        """Test delivery filtering"""
        result_with_filter = self.recommender.recommend_by_product(
            "P001",
            {'location': 'Addis Ababa', 'filter_delivery': True, 'top_n': 10}
        )
        
        # Count products with delivery available
        delivery_count = sum(
            1 for rec in result_with_filter['recommendations'] 
            if rec.get('delivery_available') == 'Yes'
        )
        
        # Should have some products with delivery
        self.assertGreater(delivery_count, 0)

class TestDataPreprocessing(unittest.TestCase):
    
    def test_data_loading(self):
        """Test data loading from CSV"""
        preprocessor = EthiopianProductPreprocessor()
        df = preprocessor.load_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        required_columns = ['product_id', 'name', 'price', 'category']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_text_cleaning(self):
        """Test Amharic and English text cleaning"""
        preprocessor = EthiopianProductPreprocessor()
        
        # Test English text
        english_text = "High-quality Samsung smartphone!"
        cleaned = preprocessor.clean_text(english_text)
        self.assertNotIn('!', cleaned)
        
        # Test Amharic text
        amharic_text = "ከፍተኛ ጥራት ያለው ስልክ!"
        cleaned = preprocessor.clean_text(amharic_text)
        self.assertIn('ከፍተኛ', cleaned)

if __name__ == '__main__':
    unittest.main()