# create train_large_dataset.py
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import EthiopianProductPreprocessor
from model_training import EthiopianRecommenderModel

def main():
    print("=" * 60)
    print("ETHIOPIAN E-COMMERCE RECOMMENDER - LARGE DATASET TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Preprocess data
    print("\n1. Preprocessing data...")
    preprocessor = EthiopianProductPreprocessor()
    df = preprocessor.load_data()
    df_processed = preprocessor.preprocess_data(df)
    preprocessor.save_processed_data(df_processed)
    
    print(f"Time for preprocessing: {time.time() - start_time:.2f} seconds")
    
    # Step 2: Train model
    print("\n2. Training recommendation model...")
    training_start = time.time()
    
    model = EthiopianRecommenderModel()
    model.train()
    model.save_model()
    
    print(f"Time for training: {time.time() - training_start:.2f} seconds")
    
    # Step 3: Test the model
    print("\n3. Testing model with sample products...")
    
    # Get sample product IDs from different categories
    sample_products = []
    for category in ['Electronics', 'Food & Beverages', 'Fashion']:
        category_products = model.df[model.df['category'] == category]
        if len(category_products) > 0:
            sample_products.append(category_products.iloc[0]['product_id'])
    
    for product_id in sample_products[:3]:  # Test only 3
        print(f"\nRecommendations for {product_id}:")
        recommendations = model.get_recommendations(product_id, top_n=3)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['name'][:50]}...")
                print(f"     Price: {rec['price']} ETB, Similarity: {rec['similarity_score']:.3f}")
        else:
            print(f"  No recommendations found for {product_id}")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Products processed: {len(model.df):,}")
    print("=" * 60)

if __name__ == "__main__":
    main()