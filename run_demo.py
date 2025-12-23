#!/usr/bin/env python3
"""
Script to launch the Ethiopian E-commerce Recommender demo
"""
import os
import sys
import subprocess
import webbrowser
import time

def main():
    print("=" * 60)
    print("ETHIOPIAN E-COMMERCE RECOMMENDER - DEMO LAUNCHER")
    print("=" * 60)
    
    # Check if model is trained
    model_path = "data/models/tfidf_model.pkl"
    if not os.path.exists(model_path):
        print("\n⚠️  Model not found! Training model first...")
        print("\nTraining the recommendation model...")
        
        # Import and run training
        sys.path.append('src')
        from model_training import EthiopianRecommenderModel
        
        model = EthiopianRecommenderModel()
        model.train()
        model.save_model()
        
        print("✅ Model training completed!")
    
    # Launch Flask app
    print("\n🚀 Launching web application...")
    print("📍 The app will be available at: http://localhost:5000")
    print("📱 Open your browser to view the interactive demo")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Open browser after a short delay
    time.sleep(2)
    webbrowser.open('http://localhost:5000')
    
    # Run the Flask app
    os.chdir('demo')
    subprocess.run(['python', 'app.py'])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")