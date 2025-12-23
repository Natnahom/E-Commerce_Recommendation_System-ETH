# create generate_dataset.py
import pandas as pd
import numpy as np
import random
from faker import Faker
import sys

def generate_realistic_ethiopian_products(num_products=10000):
    """Generate realistic Ethiopian e-commerce products"""
    fake = Faker()
    
    print(f"Generating {num_products:,} Ethiopian products...")
    
    # Ethiopian-specific data
    locations = ['Addis Ababa', 'Bahir Dar', 'Hawassa', 'Mekelle', 'Dire Dawa', 'Jimma', 
                 'Gondar', 'Adama', 'Arba Minch', 'Asosa', 'Jijiga', 'Harar']
    categories = {
        'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Televisions', 'Accessories', 
                       'Headphones', 'Speakers', 'Cameras', 'Watches', 'Chargers'],
        'Food & Beverages': ['Coffee', 'Tea', 'Spices', 'Grains', 'Honey', 'Oil', 
                           'Flour', 'Sugar', 'Salt', 'Pasta', 'Beverages'],
        'Fashion': ['Traditional Wear', 'Modern Clothing', 'Shoes', 'Accessories', 
                   'Bags', 'Jewelry', 'Watches', 'Belts'],
        'Home & Kitchen': ['Cookware', 'Appliances', 'Furniture', 'Decor', 
                          'Bedding', 'Lighting', 'Storage', 'Cleaning'],
        'Sports': ['Equipment', 'Sportswear', 'Footwear', 'Fitness', 'Outdoor'],
        'Beauty & Personal Care': ['Skincare', 'Haircare', 'Makeup', 'Fragrances', 
                                  'Personal Hygiene', 'Shaving'],
        'Books & Stationery': ['Fiction', 'Educational', 'Religion', 'Children', 
                              'Business', 'Stationery', 'Art Supplies'],
        'Health': ['Medicines', 'Supplements', 'First Aid', 'Medical Equipment'],
        'Automotive': ['Parts', 'Accessories', 'Tools', 'Car Care'],
        'Baby & Kids': ['Clothing', 'Toys', 'Feeding', 'Furniture', 'Safety']
    }
    
    brands = {
        'Electronics': ['Samsung', 'TECNO', 'Infinix', 'Hisense', 'Dell', 'HP', 'Lenovo', 
                       'Apple', 'Huawei', 'Xiaomi', 'Anker', 'Sony', 'Canon', 'Nikon'],
        'Food & Beverages': ['Ethio Coffee', 'Tomoca', 'Kaldi\'s', 'Green Farms', 
                           'Sheba Honey', 'Awash', 'Dashen', 'Harar', 'Yirgacheffe'],
        'Fashion': ['Sheba Crafts', 'Habesha Design', 'Lucy Fashion', 'Tibeb Athletics',
                   'Mekelle Leather', 'Gonder Textiles', 'Awraja'],
        'Home & Kitchen': ['Ethio Pottery', 'Mega Mitad', 'Addis Home', 'Blue Nile Crafts',
                          'Sheger Furniture', 'Nile Decor', 'Tana Appliances'],
        'Sports': ['Adidas', 'Nike', 'Puma', 'Reebok', 'Ethio Sports', 'Runner\'s Choice'],
        'Beauty & Personal Care': ['Nivea', 'Dove', 'L\'Oréal', 'Pantene', 'Ethio Herbal',
                                  'Sheba Beauty', 'Natural Touch'],
        'Books & Stationery': ['EPHI Press', 'Addis Books', 'Unity Publishers', 
                              'Mega Stationery', 'Student Choice'],
        'Health': ['Pharma Ethiopia', 'MediCare', 'Health Plus', 'Natural Remedies'],
        'Automotive': ['Auto Addis', 'Car Care Ethiopia', 'Mekina Parts', 'Drive Safe'],
        'Baby & Kids': ['Baby Ethiopia', 'Kids World', 'Tiny Steps', 'Happy Child']
    }
    
    # Product name templates by category
    product_templates = {
        'Electronics': {
            'Smartphones': ['{brand} {model} {number}', '{brand} {model} Pro', 
                           '{brand} {model} Lite', '{brand} {model} Max'],
            'Laptops': ['{brand} {model} Laptop', '{brand} {series} Notebook',
                       '{brand} {model} Ultrabook'],
            'Televisions': ['{brand} {size}" Smart TV', '{brand} {size}" LED TV',
                           '{brand} {size}" UHD TV'],
        },
        'Food & Beverages': {
            'Coffee': ['Ethiopian {type} Coffee Beans', '{brand} {type} Coffee',
                      'Premium {type} Coffee {weight}'],
            'Tea': ['{brand} Ethiopian Tea', 'Premium Tea Leaves {weight}',
                   'Traditional Ethiopian Tea'],
        },
        'Fashion': {
            'Traditional Wear': ['{brand} Habesha Kemis', 'Ethiopian {item}',
                               'Traditional {material} {item}'],
            'Modern Clothing': ['{brand} {type} Shirt', '{brand} {style} Dress',
                              '{brand} Casual Wear'],
        }
    }
    
    # Models and series
    phone_models = ['Galaxy', 'Spark', 'Hot', 'Note', 'Pro', 'Lite', 'Max', 'Prime']
    laptop_series = ['Inspiron', 'Pavilion', 'ThinkPad', 'Ideapad', 'Vostro', 'EliteBook']
    coffee_types = ['Yirgacheffe', 'Sidamo', 'Harrar', 'Limu', 'Jimma', 'Kaffa']
    
    products = []
    batch_size = 1000
    
    for i in range(num_products):
        if i % batch_size == 0:
            print(f"Generated {i:,} products...")
            sys.stdout.flush()
        
        product_id = f'ETP{i+1:05d}'
        
        # Randomly select category and subcategory
        category = random.choice(list(categories.keys()))
        subcategory = random.choice(categories[category])
        
        # Select brand based on category
        brand = random.choice(brands.get(category, ['Ethio Brand']))
        
        # Generate product name
        if category in product_templates and subcategory in product_templates[category]:
            template = random.choice(product_templates[category][subcategory])
            if category == 'Electronics':
                if subcategory == 'Smartphones':
                    name = template.format(brand=brand, model=random.choice(phone_models), 
                                         number=random.randint(5, 20))
                elif subcategory == 'Laptops':
                    name = template.format(brand=brand, model=random.choice(laptop_series),
                                         series=random.choice(['A', 'B', 'C', 'X', 'Y']))
                elif subcategory == 'Televisions':
                    name = template.format(brand=brand, size=random.choice([32, 40, 43, 50, 55, 65]))
            elif category == 'Food & Beverages':
                if subcategory == 'Coffee':
                    name = template.format(brand=brand, type=random.choice(coffee_types),
                                         weight=random.choice(['250g', '500g', '1kg']))
                else:
                    name = template.format(brand=brand, weight=random.choice(['100g', '250g', '500g']))
            else:
                name = template.format(brand=brand, item=subcategory, type=random.choice(['Casual', 'Formal']),
                                     material=random.choice(['Cotton', 'Silk', 'Wool']),
                                     style=random.choice(['Modern', 'Traditional', 'Casual']))
        else:
            name = f'{brand} {subcategory} {random.choice(["Premium", "Standard", "Deluxe"])}'
        
        # Generate Amharic name
        amharic_prefixes = {
            'Samsung': 'ሳምሰንግ',
            'TECNO': 'ቴክኖ',
            'Infinix': 'ኢንፊኒክስ',
            'Hisense': 'ሃይሰንስ',
            'Dell': 'ዴል',
            'HP': 'ኤችፒ',
            'Ethio Coffee': 'ኢትዮጵያዊ ቡና',
            'Sheba Crafts': 'ሸባ ስራ',
            'Habesha Design': 'ሐበሻ ዲዛይን'
        }
        
        amharic_brand = amharic_prefixes.get(brand, brand)
        name_amharic = f'{amharic_brand} {subcategory}'
        
        # Generate description
        descriptions = [
            f'High quality {subcategory.lower()} from Ethiopia. {fake.sentence()}',
            f'Premium {subcategory.lower()} made with Ethiopian craftsmanship. {fake.sentence()}',
            f'Authentic Ethiopian {subcategory.lower()}. {fake.sentence()}',
            f'Best quality {subcategory.lower()} sourced locally. {fake.sentence()}'
        ]
        description = random.choice(descriptions)
        
        # Amharic description
        amharic_descriptions = [
            f'ከፍተኛ ጥራት ያለው ኢትዮጵያዊ {subcategory.lower()}. ይህ ምርት ለእርስዎ ተስማሚ ነው።',
            f'በኢትዮጵያ የተሰራ የፕሪሚየም ጥራት {subcategory.lower()}። እርስዎ የሚፈልጉትን ሁሉ ይህ ያቀርባል።',
            f'እውነተኛ ኢትዮጵያዊ {subcategory.lower()}። ጥራቱ የተረጋገጠ ነው።'
        ]
        description_amharic = random.choice(amharic_descriptions)
        
        # Generate price based on category
        price_ranges = {
            'Electronics': (1500, 150000),
            'Food & Beverages': (50, 10000),
            'Fashion': (300, 25000),
            'Home & Kitchen': (200, 50000),
            'Sports': (500, 30000),
            'Beauty & Personal Care': (100, 15000),
            'Books & Stationery': (80, 8000),
            'Health': (200, 20000),
            'Automotive': (500, 100000),
            'Baby & Kids': (150, 20000)
        }
        
        min_price, max_price = price_ranges.get(category, (100, 10000))
        price = random.randint(min_price, max_price)
        
        # Round price to nearest 50 or 100
        if price > 1000:
            price = round(price / 100) * 100
        else:
            price = round(price / 50) * 50
        
        # Other fields
        location = random.choice(locations)
        delivery_available = random.choice(['Yes', 'No', 'Yes', 'Yes'])  # 75% yes
        rating = round(random.uniform(3.0, 5.0), 1)
        
        # Stock status based on probability
        stock_prob = random.random()
        if stock_prob < 0.7:  # 70% in stock
            stock_status = 'In Stock'
            stock_quantity = random.randint(1, 500)
        elif stock_prob < 0.9:  # 20% low stock
            stock_status = 'Low Stock'
            stock_quantity = random.randint(1, 20)
        else:  # 10% out of stock
            stock_status = 'Out of Stock'
            stock_quantity = 0
        
        # Generate tags
        tags = f'{category},{subcategory},{location},{brand},Ethiopian-made'
        if random.random() < 0.3:
            tags += ',Premium'
        if random.random() < 0.2:
            tags += ',New'
        if random.random() < 0.4:
            tags += ',Best-Seller'
        
        product = {
            'product_id': product_id,
            'name': name,
            'name_amharic': name_amharic,
            'description': description,
            'description_amharic': description_amharic,
            'category': category,
            'subcategory': subcategory,
            'price': price,
            'currency': 'ETB',
            'brand': brand,
            'location': location,
            'delivery_available': delivery_available,
            'rating': rating,
            'stock_status': stock_status,
            'stock_quantity': stock_quantity,
            'weight_kg': round(random.uniform(0.1, 50.0), 2),
            'dimensions': f'{random.randint(5, 200)}x{random.randint(5, 200)}x{random.randint(1, 100)} cm',
            'warranty_months': random.choice([0, 3, 6, 12, 24]),
            'seller_name': f'{brand} {random.choice(["Store", "Shop", "Mart", "Center", "Hub"])}',
            'seller_rating': round(random.uniform(3.0, 5.0), 1),
            'tags': tags
        }
        
        products.append(product)
    
    return pd.DataFrame(products)

def save_dataset_in_chunks(df, base_filename, chunk_size=2000):
    """Save large dataset in chunks"""
    print(f"\nSaving {len(df):,} products to {base_filename}...")
    
    # Save all data in one file
    df.to_csv(f'{base_filename}.csv', index=False, encoding='utf-8')
    print(f"Saved complete dataset to {base_filename}.csv")
    
    # Also save a sample for testing
    sample_size = min(100, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    df_sample.to_csv(f'{base_filename}_sample.csv', index=False, encoding='utf-8')
    print(f"Saved {sample_size} sample products to {base_filename}_sample.csv")

if __name__ == "__main__":
    # Generate 10,000 realistic products
    print("=" * 60)
    print("ETHIOPIAN E-COMMERCE PRODUCT DATASET GENERATOR")
    print("=" * 60)
    
    df = generate_realistic_ethiopian_products(10000)
    
    # Save dataset
    save_dataset_in_chunks(df, 'data/raw/ethiopian_products_10k')
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"Total Products: {len(df):,}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Average Price: {df['price'].mean():.2f} ETB")
    print(f"Products with Delivery: {(df['delivery_available'] == 'Yes').sum()}")
    print(f"Average Rating: {df['rating'].mean():.2f}")
    
    print("\nCategory Distribution:")
    print(df['category'].value_counts())