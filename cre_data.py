# create generate_dataset.py
import pandas as pd
import numpy as np
import random
from faker import Faker
import sys
from datetime import datetime

def generate_realistic_ethiopian_products(num_products=10000):
    """Generate realistic Ethiopian e-commerce products"""
    fake = Faker()
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {num_products:,} Ethiopian products...")
    
    # Ethiopian-specific data - using actual Ethiopian cities
    locations = {
        'Addis Ababa': 0.35,  # Capital city - highest probability
        'Bahir Dar': 0.08,
        'Hawassa': 0.08,
        'Mekelle': 0.08,
        'Dire Dawa': 0.07,
        'Jimma': 0.07,
        'Gondar': 0.07,
        'Adama': 0.07,
        'Arba Minch': 0.04,
        'Asosa': 0.03,
        'Jijiga': 0.03,
        'Harar': 0.03
    }
    
    # Categories with realistic Ethiopian market distribution
    categories = {
        'Food & Beverages': 0.25,  # Largest category in Ethiopian market
        'Fashion': 0.18,
        'Beauty & Personal Care': 0.15,
        'Electronics': 0.12,
        'Home & Kitchen': 0.10,
        'Books & Stationery': 0.06,
        'Health': 0.05,
        'Sports': 0.04,
        'Baby & Kids': 0.03,
        'Automotive': 0.02
    }
    
    # Subcategories with realistic relationships
    subcategories = {
        'Food & Beverages': {
            'Coffee': 0.30, 'Tea': 0.15, 'Spices': 0.15, 'Grains': 0.12,
            'Honey': 0.08, 'Oil': 0.06, 'Flour': 0.05, 'Sugar': 0.04,
            'Salt': 0.03, 'Pasta': 0.01, 'Beverages': 0.01
        },
        'Fashion': {
            'Traditional Wear': 0.35, 'Modern Clothing': 0.25, 'Shoes': 0.15,
            'Accessories': 0.10, 'Bags': 0.07, 'Belts': 0.05, 'Jewelry': 0.03
        },
        'Beauty & Personal Care': {
            'Haircare': 0.25, 'Skincare': 0.20, 'Makeup': 0.20, 'Fragrances': 0.15,
            'Personal Hygiene': 0.10, 'Shaving': 0.10
        },
        'Electronics': {
            'Smartphones': 0.40, 'Laptops': 0.15, 'Televisions': 0.12,
            'Headphones': 0.10, 'Speakers': 0.08, 'Accessories': 0.06,
            'Cameras': 0.04, 'Watches': 0.03, 'Chargers': 0.02
        },
        'Home & Kitchen': {
            'Cookware': 0.25, 'Furniture': 0.20, 'Bedding': 0.15, 'Appliances': 0.12,
            'Cleaning': 0.10, 'Storage': 0.08, 'Lighting': 0.06, 'Decor': 0.04
        }
    }
    
    # Brands with realistic Ethiopian market presence
    brands = {
        'Electronics': {
            'TECNO': 0.25, 'Infinix': 0.20, 'Samsung': 0.15, 'Hisense': 0.10,
            'Huawei': 0.08, 'Dell': 0.06, 'HP': 0.05, 'Lenovo': 0.04,
            'Apple': 0.03, 'Xiaomi': 0.02, 'Anker': 0.01, 'Canon': 0.01
        },
        'Food & Beverages': {
            'Ethio Coffee': 0.25, 'Tomoca': 0.15, 'Kaldi\'s': 0.15, 'Dashen': 0.12,
            'Awash': 0.10, 'Sheba Honey': 0.08, 'Harar': 0.07, 'Yirgacheffe': 0.05,
            'Green Farms': 0.03
        },
        'Fashion': {
            'Habesha Design': 0.20, 'Sheba Crafts': 0.18, 'Lucy Fashion': 0.16,
            'Tibeb Athletics': 0.14, 'Mekelle Leather': 0.12, 'Gonder Textiles': 0.10,
            'Awraja': 0.10
        },
        'Beauty & Personal Care': {
            'Ethio Herbal': 0.25, 'Sheba Beauty': 0.20, 'Nivea': 0.15, 'Dove': 0.12,
            'Pantene': 0.10, 'L\'Oréal': 0.08, 'Natural Touch': 0.06,
            'Natural Remedies': 0.04
        },
        'Home & Kitchen': {
            'Mega Mitad': 0.25, 'Ethio Pottery': 0.20, 'Addis Home': 0.18,
            'Sheger Furniture': 0.15, 'Blue Nile Crafts': 0.12, 'Nile Decor': 0.06,
            'Tana Appliances': 0.04
        }
    }
    
    # Product models and variants
    phone_models = {
        'TECNO': ['Spark', 'Camon', 'Pova', 'Phantom'],
        'Infinix': ['Hot', 'Note', 'Zero', 'Smart'],
        'Samsung': ['Galaxy A', 'Galaxy M', 'Galaxy S'],
        'Huawei': ['P', 'Y', 'Nova']
    }
    
    laptop_models = {
        'Dell': ['Inspiron', 'Vostro', 'XPS'],
        'HP': ['Pavilion', 'EliteBook', 'ProBook'],
        'Lenovo': ['ThinkPad', 'Ideapad']
    }
    
    coffee_types = ['Yirgacheffe', 'Sidamo', 'Harrar', 'Limu', 'Jimma', 'Kaffa']
    
    # Price ranges with realistic Ethiopian market prices (in ETB)
    price_ranges = {
        'Electronics': {
            'Smartphones': (3000, 50000),
            'Laptops': (15000, 120000),
            'Televisions': (5000, 100000),
            'Headphones': (500, 15000),
            'Speakers': (800, 25000)
        },
        'Food & Beverages': {
            'Coffee': (150, 5000),
            'Tea': (50, 1500),
            'Spices': (100, 3000),
            'Grains': (80, 2500),
            'Honey': (200, 4000)
        },
        'Fashion': {
            'Traditional Wear': (800, 20000),
            'Modern Clothing': (500, 15000),
            'Shoes': (600, 10000),
            'Accessories': (200, 5000)
        },
        'Beauty & Personal Care': {
            'Haircare': (150, 3000),
            'Skincare': (200, 4000),
            'Makeup': (300, 6000),
            'Fragrances': (500, 8000)
        }
    }
    
    # Realistic Ethiopian seller names
    seller_suffixes = {
        'Electronics': ['Tech Hub', 'Electronics Store', 'Digital Mart', 'Gadget Center'],
        'Food & Beverages': ['Store', 'Market', 'Supplier', 'Distributor'],
        'Fashion': ['Boutique', 'Fashion House', 'Design Studio', 'Collection'],
        'Beauty & Personal Care': ['Beauty Shop', 'Cosmetics Store', 'Care Center']
    }
    
    products = []
    
    for i in range(num_products):
        if i % 1000 == 0:
            print(f"Generated {i:,} products...")
            sys.stdout.flush()
        
        product_id = f'ETP{i+1:05d}'
        
        # Select category based on probability distribution
        category = random.choices(
            list(categories.keys()), 
            weights=list(categories.values())
        )[0]
        
        # Select subcategory
        if category in subcategories:
            subcategory = random.choices(
                list(subcategories[category].keys()),
                weights=list(subcategories[category].values())
            )[0]
        else:
            # Fallback subcategories for remaining categories
            fallback_subs = {
                'Books & Stationery': ['Fiction', 'Educational', 'Religion', 'Children', 'Business'],
                'Health': ['Medicines', 'Supplements', 'First Aid', 'Medical Equipment'],
                'Sports': ['Equipment', 'Sportswear', 'Footwear', 'Fitness'],
                'Baby & Kids': ['Clothing', 'Toys', 'Feeding', 'Furniture'],
                'Automotive': ['Parts', 'Accessories', 'Tools', 'Car Care']
            }
            subcategory = random.choice(fallback_subs.get(category, ['Standard']))
        
        # Select brand
        if category in brands:
            brand_list = list(brands[category].keys())
            brand_weights = list(brands[category].values())
            brand = random.choices(brand_list, weights=brand_weights)[0]
        else:
            # Fallback brands
            fallback_brands = {
                'Books & Stationery': ['Unity Publishers', 'EPHI Press', 'Addis Books'],
                'Health': ['Pharma Ethiopia', 'MediCare', 'Health Plus'],
                'Sports': ['Ethio Sports', 'Runner\'s Choice', 'Tibeb Athletics'],
                'Baby & Kids': ['Baby Ethiopia', 'Kids World', 'Happy Child'],
                'Automotive': ['Auto Addis', 'Mekina Parts', 'Drive Safe']
            }
            brand = random.choice(fallback_brands.get(category, ['Ethio Brand']))
        
        # Generate realistic product name
        name = generate_product_name(category, subcategory, brand, phone_models, laptop_models, coffee_types)
        
        # Generate Amharic name
        name_amharic = generate_amharic_name(category, subcategory, brand)
        
        # Generate realistic descriptions
        description = generate_description(category, subcategory, brand, fake)
        description_amharic = generate_amharic_description(category, subcategory)
        
        # Generate realistic price
        price = generate_price(category, subcategory, price_ranges)
        
        # Select location based on probability
        location = random.choices(
            list(locations.keys()), 
            weights=list(locations.values())
        )[0]
        
        # Delivery availability (more likely in major cities)
        major_cities = ['Addis Ababa', 'Bahir Dar', 'Hawassa', 'Mekelle', 'Dire Dawa']
        if location in major_cities:
            delivery_available = random.choices(['Yes', 'No'], weights=[0.85, 0.15])[0]
        else:
            delivery_available = random.choices(['Yes', 'No'], weights=[0.70, 0.30])[0]
        
        # Generate rating with realistic distribution
        rating = generate_realistic_rating(brand)
        
        # Stock status with realistic patterns
        stock_status, stock_quantity = generate_stock_status(brand, category)
        
        # Generate realistic weight and dimensions
        weight_kg = generate_realistic_weight(category, subcategory)
        dimensions = generate_realistic_dimensions(category, subcategory)
        
        # Warranty (electronics get more warranty)
        if category == 'Electronics':
            warranty_months = random.choices([6, 12, 24], weights=[0.4, 0.4, 0.2])[0]
        else:
            warranty_months = random.choices([0, 3, 6], weights=[0.6, 0.3, 0.1])[0]
        
        # Seller name
        seller_name = generate_seller_name(category, brand, seller_suffixes)
        
        # Seller rating (related to product rating)
        seller_rating = min(5.0, max(2.5, rating + random.uniform(-0.5, 0.3)))
        seller_rating = round(seller_rating, 1)
        
        # Tags
        tags = generate_tags(category, subcategory, location, brand)
        
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
            'weight_kg': weight_kg,
            'dimensions': dimensions,
            'warranty_months': warranty_months,
            'seller_name': seller_name,
            'seller_rating': seller_rating,
            'tags': tags
        }
        
        products.append(product)
    
    return pd.DataFrame(products)

def generate_product_name(category, subcategory, brand, phone_models, laptop_models, coffee_types):
    """Generate realistic product names"""
    if category == 'Electronics':
        if subcategory == 'Smartphones':
            if brand in phone_models:
                model = random.choice(phone_models[brand])
                number = random.choice(['10', '11', '12', '13', '20', '30', 'Pro', 'Lite', 'Max'])
                return f'{brand} {model} {number}'
            else:
                return f'{brand} Smartphone'
        
        elif subcategory == 'Laptops':
            if brand in laptop_models:
                series = random.choice(laptop_models[brand])
                model_num = random.choice(['15', '14', '13', 'X360', 'G5'])
                return f'{brand} {series} {model_num}'
            else:
                return f'{brand} Laptop'
        
        elif subcategory == 'Televisions':
            sizes = ['32"', '40"', '43"', '50"', '55"', '65"']
            types = ['LED TV', 'Smart TV', 'UHD TV', 'Android TV']
            return f'{brand} {random.choice(sizes)} {random.choice(types)}'
    
    elif category == 'Food & Beverages':
        if subcategory == 'Coffee':
            coffee_type = random.choice(coffee_types)
            weight = random.choice(['250g', '500g', '1kg'])
            return f'Ethiopian {coffee_type} Coffee {weight}'
        
        elif subcategory == 'Tea':
            types = ['Black', 'Green', 'Herbal', 'Traditional']
            weight = random.choice(['100g', '250g', '500g'])
            return f'{brand} {random.choice(types)} Tea {weight}'
    
    elif category == 'Fashion':
        if subcategory == 'Traditional Wear':
            items = ['Habesha Kemis', 'Kuta', 'Netela', 'Shawl', 'Dress']
            return f'{brand} {random.choice(items)}'
        
        elif subcategory == 'Modern Clothing':
            types = ['T-Shirt', 'Shirt', 'Dress', 'Pants', 'Jacket']
            return f'{brand} {random.choice(types)}'
    
    # Default naming pattern
    variants = ['Premium', 'Standard', 'Deluxe', 'Classic', 'Professional']
    return f'{brand} {subcategory} {random.choice(variants)}'

def generate_amharic_name(category, subcategory, brand):
    """Generate Amharic product names"""
    amharic_brands = {
        'TECNO': 'ቴክኖ',
        'Infinix': 'ኢንፊኒክስ',
        'Samsung': 'ሳምሰንግ',
        'Hisense': 'ሃይሰንስ',
        'Dell': 'ዴል',
        'Ethio Coffee': 'ኢትዮጵያዊ ቡና',
        'Sheba Crafts': 'ሸባ ስራ',
        'Habesha Design': 'ሐበሻ ዲዛይን',
        'Ethio Herbal': 'ኢትዮጵያዊ ሐምራዊ',
        'Mekelle Leather': 'መቀሌ ልብስ',
        'Tomoca': 'ቶሞካ',
        'Kaldi\'s': 'ካልዲስ',
        'Dashen': 'ዳሸን',
        'Awash': 'አዋሽ',
        'Harar': 'ሐረር',
        'Yirgacheffe': 'ይርጋጨፍ',
        'Lucy Fashion': 'ሉሲ ፋሽን',
        'Tibeb Athletics': 'ጥበብ አትሌቲክስ',
        'Gonder Textiles': 'ጎንደር ጨርቃ ጨርቅ',
        'Awraja': 'አውራጃ'
    }
    
    amharic_subcategories = {
        'Smartphones': 'ስማርትፎን',
        'Coffee': 'ቡና',
        'Tea': 'ሻይ',
        'Traditional Wear': 'ባህላዊ ልብስ',
        'Modern Clothing': 'ዘመናዊ ልብስ',
        'Shoes': 'ጫማ',
        'Accessories': 'ተጨማሪ ዕቃዎች',
        'Haircare': 'ፀጉር እንክብካቤ',
        'Skincare': 'ቆዳ እንክብካቤ',
        'Makeup': 'ማጅ',
        'Fragrances': 'ሽቶ',
        'Laptops': 'ላፕቶፕ',
        'Televisions': 'ቴሌቪዥን',
        'Headphones': 'ሄድፎን',
        'Speakers': 'ስፒከር',
        'Cameras': 'ካሜራ',
        'Watches': 'ሰዓት'
    }
    
    brand_am = amharic_brands.get(brand, brand)
    subcategory_am = amharic_subcategories.get(subcategory, subcategory)
    
    return f'{brand_am} {subcategory_am}'

def generate_description(category, subcategory, brand, fake):
    """Generate realistic descriptions"""
    if category == 'Electronics':
        descs = [
            f'{brand} {subcategory} with premium features. {fake.sentence()}',
            f'High-performance {subcategory.lower()} from {brand}. Features include {fake.word()} and {fake.word()}.',
            f'{brand}\'s latest {subcategory.lower()} model with advanced technology.'
        ]
    
    elif category == 'Food & Beverages':
        descs = [
            f'Authentic Ethiopian {subcategory.lower()} sourced from local farmers. {fake.sentence()}',
            f'Premium quality {subcategory.lower()} harvested in Ethiopia. {fake.sentence()}',
            f'Traditional Ethiopian {subcategory.lower()} with rich flavor and aroma.'
        ]
    
    elif category == 'Fashion':
        descs = [
            f'Beautiful {subcategory.lower()} made with traditional Ethiopian craftsmanship.',
            f'Elegant {subcategory.lower()} from {brand}, perfect for special occasions.',
            f'Handcrafted {subcategory.lower()} using premium materials.'
        ]
    
    else:
        descs = [
            f'High quality {subcategory.lower()} from Ethiopia. {fake.sentence()}',
            f'Premium {subcategory.lower()} suitable for daily use. {fake.sentence()}',
            f'Reliable {subcategory.lower()} with excellent performance.'
        ]
    
    return random.choice(descs)

def generate_amharic_description(category, subcategory):
    """Generate Amharic descriptions"""
    descriptions = {
        'Electronics': [
            'ከፍተኛ ጥራት ያለው የኤሌክትሮኒክስ ምርት። ዘመናዊ ቴክኖሎጂ እና ከፍተኛ አፈጻጸም።',
            'በኢትዮጵያ የሚገኝ የኤሌክትሮኒክስ ምርት። ረጅም ጊዜ ለመጠቀም የተሰራ።'
        ],
        'Food & Beverages': [
            'ከፍተኛ ጥራት ያለው ኢትዮጵያዊ ምርት። ንጹህ እና ጤናማ።',
            'በኢትዮጵያ ከሚገኙ ሰብሎች የተሰራ ንጹህ ምርት።'
        ],
        'Fashion': [
            'በኢትዮጵያ የተሰራ የልብስ ምርት። ባህላዊ ዲዛይን እና ከፍተኛ ጥራት።',
            'ኢትዮጵያዊ ባህልን የሚያንፀባርቅ ልብስ። ለተለያዩ ክስተቶች ተስማሚ።'
        ],
        'Beauty & Personal Care': [
            'ንጹህ እና ደህንነቱ የተረጋገጠ የእንክብካቤ ምርት።',
            'በተፈጥሮ ንጥረ ነገሮች የተሰራ እንክብካቤ ምርት።'
        ]
    }
    
    if category in descriptions:
        return random.choice(descriptions[category])
    else:
        return 'ከፍተኛ ጥራት ያለው ኢትዮጵያዊ ምርት። ለእርስዎ ተስማሚ ነው።'

def generate_price(category, subcategory, price_ranges):
    """Generate realistic prices"""
    if category in price_ranges and subcategory in price_ranges[category]:
        min_price, max_price = price_ranges[category][subcategory]
        price = random.randint(min_price, max_price)
        
        # Round to realistic increments
        if price < 1000:
            price = round(price / 10) * 10
        elif price < 10000:
            price = round(price / 100) * 100
        else:
            price = round(price / 500) * 500
        
        return price
    
    # Fallback price ranges
    fallback_ranges = {
        'Books & Stationery': (100, 3000),
        'Health': (200, 5000),
        'Sports': (500, 15000),
        'Baby & Kids': (300, 8000),
        'Automotive': (1000, 25000)
    }
    
    if category in fallback_ranges:
        min_price, max_price = fallback_ranges[category]
        price = random.randint(min_price, max_price)
        return round(price / 100) * 100
    
    # Default price range
    return random.randint(500, 10000)

def generate_realistic_rating(brand):
    """Generate realistic product ratings"""
    # Premium brands tend to have higher ratings
    premium_brands = ['Apple', 'Samsung', 'Dell', 'HP', 'Lenovo', 'Ethio Coffee', 'Habesha Design']
    
    if brand in premium_brands:
        base_rating = 4.0
    else:
        base_rating = 3.5
    
    rating = base_rating + random.uniform(0, 0.8)
    rating = min(5.0, max(2.5, rating))  # Keep within 2.5-5.0 range
    return round(rating, 1)

def generate_stock_status(brand, category):
    """Generate realistic stock status"""
    # Certain categories/brands are more likely to be in stock
    if category in ['Electronics', 'Food & Beverages']:
        in_stock_prob = 0.75
    else:
        in_stock_prob = 0.65
    
    if random.random() < in_stock_prob:
        stock_status = 'In Stock'
        stock_quantity = random.randint(5, 500)
        
        # Popular brands might have more stock
        popular_brands = ['TECNO', 'Infinix', 'Ethio Coffee', 'Nivea', 'Dove']
        if brand in popular_brands:
            stock_quantity = random.randint(20, 1000)
    
    elif random.random() < 0.7:  # 70% of remaining are low stock
        stock_status = 'Low Stock'
        stock_quantity = random.randint(1, 10)
    else:
        stock_status = 'Out of Stock'
        stock_quantity = 0
    
    return stock_status, stock_quantity

def generate_realistic_weight(category, subcategory):
    """Generate realistic weight in kg"""
    if category == 'Electronics':
        if subcategory == 'Smartphones':
            return round(random.uniform(0.15, 0.25), 2)
        elif subcategory == 'Laptops':
            return round(random.uniform(1.2, 2.5), 2)
        elif subcategory == 'Televisions':
            return round(random.uniform(5.0, 25.0), 2)
        elif subcategory == 'Headphones':
            return round(random.uniform(0.1, 0.5), 2)
        elif subcategory == 'Speakers':
            return round(random.uniform(0.5, 5.0), 2)
    
    elif category == 'Food & Beverages':
        if subcategory in ['Coffee', 'Tea', 'Spices']:
            return round(random.uniform(0.1, 1.0), 2)
        elif subcategory in ['Grains', 'Flour', 'Sugar']:
            return round(random.uniform(0.5, 5.0), 2)
        elif subcategory == 'Honey':
            return round(random.uniform(0.3, 2.0), 2)
        elif subcategory == 'Oil':
            return round(random.uniform(0.5, 3.0), 2)
    
    elif category == 'Fashion':
        if subcategory == 'Shoes':
            return round(random.uniform(0.5, 1.5), 2)
        elif subcategory == 'Traditional Wear':
            return round(random.uniform(0.3, 1.2), 2)
        elif subcategory == 'Modern Clothing':
            return round(random.uniform(0.2, 0.8), 2)
        elif subcategory == 'Bags':
            return round(random.uniform(0.3, 2.0), 2)
    
    elif category == 'Beauty & Personal Care':
        if subcategory in ['Haircare', 'Skincare', 'Makeup']:
            return round(random.uniform(0.1, 0.5), 2)
        elif subcategory == 'Fragrances':
            return round(random.uniform(0.05, 0.2), 2)
    
    # Default weights
    return round(random.uniform(0.1, 5.0), 2)

def generate_realistic_dimensions(category, subcategory):
    """Generate realistic dimensions"""
    if category == 'Electronics':
        if subcategory == 'Smartphones':
            # Convert cm to mm for smartphone dimensions, then format
            length = random.randint(140, 165)  # 14.0-16.5 cm
            width = random.randint(65, 80)     # 6.5-8.0 cm
            thickness = random.randint(7, 10)  # 0.7-1.0 cm
            return f'{length}x{width}x{thickness} cm'
        
        elif subcategory == 'Laptops':
            return f'{random.randint(30, 40)}x{random.randint(20, 25)}x{random.randint(1, 3)} cm'
        
        elif subcategory == 'Televisions':
            return f'{random.randint(70, 150)}x{random.randint(40, 90)}x{random.randint(5, 15)} cm'
        
        elif subcategory == 'Headphones':
            return f'{random.randint(15, 25)}x{random.randint(15, 20)}x{random.randint(5, 10)} cm'
        
        elif subcategory == 'Speakers':
            return f'{random.randint(10, 40)}x{random.randint(10, 30)}x{random.randint(10, 25)} cm'
    
    elif category == 'Food & Beverages':
        if subcategory in ['Coffee', 'Tea', 'Spices']:
            return f'{random.randint(10, 20)}x{random.randint(10, 15)}x{random.randint(3, 8)} cm'
        elif subcategory in ['Grains', 'Flour', 'Sugar']:
            return f'{random.randint(20, 40)}x{random.randint(15, 25)}x{random.randint(5, 10)} cm'
        elif subcategory == 'Oil':
            return f'{random.randint(8, 15)}x{random.randint(8, 15)}x{random.randint(20, 30)} cm'
    
    elif category == 'Fashion':
        if subcategory == 'Shoes':
            return f'{random.randint(25, 35)}x{random.randint(10, 15)}x{random.randint(10, 15)} cm'
        elif subcategory in ['Traditional Wear', 'Modern Clothing']:
            return f'{random.randint(40, 60)}x{random.randint(30, 50)}x{random.randint(2, 5)} cm'
        elif subcategory == 'Bags':
            return f'{random.randint(25, 50)}x{random.randint(15, 30)}x{random.randint(5, 15)} cm'
    
    elif category == 'Beauty & Personal Care':
        if subcategory in ['Haircare', 'Skincare', 'Makeup']:
            return f'{random.randint(5, 15)}x{random.randint(5, 10)}x{random.randint(5, 10)} cm'
        elif subcategory == 'Fragrances':
            return f'{random.randint(8, 12)}x{random.randint(3, 6)}x{random.randint(3, 6)} cm'
    
    # Default dimensions
    return f'{random.randint(10, 50)}x{random.randint(5, 30)}x{random.randint(2, 20)} cm'

def generate_seller_name(category, brand, seller_suffixes):
    """Generate realistic seller names"""
    if category in seller_suffixes:
        suffix = random.choice(seller_suffixes[category])
    else:
        suffix = random.choice(['Store', 'Shop', 'Supplier', 'Mart'])
    
    return f'{brand} {suffix}'

def generate_tags(category, subcategory, location, brand):
    """Generate realistic tags"""
    tags = [category, subcategory, location, brand, 'Ethiopian-made']
    
    # Add premium tag for higher probability based on brand/category
    premium_items = ['Apple', 'Samsung', 'Dell', 'Ethio Coffee', 'Habesha Design']
    if brand in premium_items or random.random() < 0.25:
        tags.append('Premium')
    
    # Add best-seller tag (30% probability)
    if random.random() < 0.30:
        tags.append('Best-Seller')
    
    # Add new tag (20% probability)
    if random.random() < 0.20:
        tags.append('New')
    
    return ','.join(tags)

def save_dataset(df, filename='ethiopian_products_10k.csv'):
    """Save the dataset"""
    print(f"\nSaving {len(df):,} products to {filename}...")
    df.to_csv(filename, index=False, encoding='utf-8')
    
    # Also save a sample
    sample_size = min(100, len(df))
    df.sample(sample_size, random_state=42).to_csv('ethiopian_products_sample.csv', index=False, encoding='utf-8')
    
    print(f"Complete dataset saved to {filename}")
    print(f"Sample of {sample_size} products saved to ethiopian_products_sample.csv")

if __name__ == "__main__":
    # Generate 10,000 realistic products
    print("=" * 60)
    print("REALISTIC ETHIOPIAN E-COMMERCE PRODUCT DATASET GENERATOR")
    print("=" * 60)
    
    df = generate_realistic_ethiopian_products(10000)
    
    # Save dataset
    save_dataset(df, 'ethiopian_products_realistic_10k.csv')
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    
    # Dataset statistics
    print("\nDATASET STATISTICS:")
    print("-" * 40)
    print(f"Total Products: {len(df):,}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Brands: {df['brand'].nunique()}")
    print(f"Locations: {df['location'].nunique()}")
    print(f"Average Price: {df['price'].mean():,.2f} ETB")
    print(f"Price Range: {df['price'].min():,.0f} - {df['price'].max():,.0f} ETB")
    print(f"Products with Delivery: {(df['delivery_available'] == 'Yes').sum():,} ({df[df['delivery_available'] == 'Yes'].shape[0]/len(df)*100:.1f}%)")
    print(f"Average Rating: {df['rating'].mean():.2f}/5.0")
    print(f"In Stock: {(df['stock_status'] == 'In Stock').sum():,} ({df[df['stock_status'] == 'In Stock'].shape[0]/len(df)*100:.1f}%)")
    
    print("\nCATEGORY DISTRIBUTION:")
    print("-" * 40)
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        avg_price = df[df['category'] == category]['price'].mean()
        print(f"{category:25} {count:5,} products ({percentage:5.1f}%) | Avg Price: {avg_price:,.0f} ETB")
    
    print("\nTOP 10 BRANDS:")
    print("-" * 40)
    brand_counts = df['brand'].value_counts().head(10)
    for brand, count in brand_counts.items():
        print(f"{brand:20} {count:5,} products")