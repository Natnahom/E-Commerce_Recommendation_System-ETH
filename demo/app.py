from flask import Flask, render_template, request, jsonify
import json
import sys
import os
# Add this import at the top
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from recommender import EthiopianEcommerceRecommender

app = Flask(__name__)
recommender = EthiopianEcommerceRecommender()

@app.route('/')
def index():
    """Home page"""
    stats = recommender.get_stats()
    categories = recommender.get_product_categories()
    
    # Get sample products for display
    sample_products = []
    for product_id in ['P001', 'P002', 'P003', 'P004', 'P005']:
        product = recommender.get_product_details(product_id)
        if product:
            sample_products.append(product)
    
    return render_template('index.html', 
                         stats=stats, 
                         categories=categories,
                         sample_products=sample_products)

# Update the recommend endpoint to handle errors
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint for recommendations"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if 'product_id' in data:
            # Product-based recommendation
            product_id = data['product_id']
            user_context = data.get('context', {})
            result = recommender.recommend_by_product(product_id, user_context)
        
        elif 'search_text' in data:
            # Search-based recommendation
            search_text = data['search_text']
            user_context = data.get('context', {})
            result = recommender.recommend_by_text_search(search_text, user_context)
        
        else:
            return jsonify({'error': 'No product_id or search_text provided'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/product/<product_id>')
def get_product(product_id):
    """Get product details"""
    product = recommender.get_product_details(product_id)
    if product:
        return jsonify(product)
    else:
        return jsonify({'error': 'Product not found'}), 404

@app.route('/demo')
def demo_page():
    """Interactive demo page"""
    return render_template('demo.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)