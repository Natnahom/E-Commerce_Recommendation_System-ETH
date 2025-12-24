## IML assignment
# Ethiopian E-commerce Product Recommender

A content-based recommendation system built for Ethiopian e-commerce platforms using unsupervised learning techniques.

## 🎯 Project Overview

This project implements a **Content-Based Filtering** recommendation system tailored for Ethiopian e-commerce markets. It addresses the cold-start problem common in emerging markets and incorporates local context like delivery feasibility and regional availability.

## 📁 Project Structure
ethiopian-ecommerce-recommender/
├── data/ # Data files
├── src/ # Source code
├── tests/ # Unit tests
├── notebooks/ # Jupyter notebooks
├── requirements.txt # Dependencies
├── config.yaml # Configuration

## 🚀 Quick Start

### 1. Installation
pip install -r requirements.txt

2. Prepare Data
Place your product CSV file in data/raw/ethiopian_products_sample.csv

3. Train Model

python run_training.py
4. Run Demo

python demo/app.py
Visit: http://localhost:5000

🔧 Features
Content-Based Filtering using TF-IDF and cosine similarity

Bilingual Support for Amharic and English

Delivery-Aware Recommendations based on location

Web Interface for interactive demo

Ethiopian Context Integration

📊 Sample Data Format
The system expects a CSV with the following columns:

product_id: Unique identifier

name: Product name (Amharic/English)

description_amharic: Description in Amharic

description_english: Description in English

category: Product category

price: Price in ETB/USD

rating: User rating (1-5)

location: Location of product

delivery_available: Yes/No


🤝 Team
This project was developed by:

- Natnahom Asfaw

- Nebiyu Ermiyas

- Nebyu Daniel

For: SWEG4112 - Introduction to Machine Learning

### **5. HOW TO RUN THE PROJECT**

#### **Step 1: Set up the environment**
# Create the folder structure
mkdir -p ethiopian-ecommerce-recommender/{data/{raw,processed,models},src,demo/{templates,static},tests,notebooks}

# Install dependencies
pip install -r requirements.txt
Step 2: Add the data
Save the CSV file as data/raw/ethiopian_products_10k.csv

Step 3: Train the model
by following train_model.ipynb

Step 4: Run the demo

by following the end of train_model.ipynb
