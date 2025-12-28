# generate_dataset_v2.py
import pandas as pd
import numpy as np
import random
from faker import Faker
import sys

# ------------------------------
# Config
# ------------------------------
NUM_PRODUCTS = 150_000
CURRENCY = 'ETB'
SEED = 42

# Keep columns EXACTLY the same as your current CSV
COLUMNS = [
    'product_id', 'name', 'name_amharic', 'description', 'description_amharic',
    'category', 'subcategory', 'price', 'currency', 'brand', 'location',
    'delivery_available', 'rating', 'stock_status', 'stock_quantity',
    'weight_kg', 'dimensions', 'warranty_months', 'seller_name',
    'seller_rating', 'tags'
]

fake = Faker()
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------
# Ethiopian context
# ------------------------------
LOCATIONS = {
    'Addis Ababa': 0.30, 'Bahir Dar': 0.08, 'Hawassa': 0.08, 'Mekelle': 0.08,
    'Dire Dawa': 0.07, 'Jimma': 0.06, 'Gondar': 0.06, 'Adama': 0.06,
    'Arba Minch': 0.04, 'Asosa': 0.03, 'Jijiga': 0.03, 'Harar': 0.03,
    'Shashemene': 0.02
}

# Dirichlet-generated probabilities to avoid fixed patterns
BASE_CATEGORIES = [
    'Food & Beverages', 'Fashion', 'Beauty & Personal Care', 'Electronics',
    'Home & Kitchen', 'Books & Stationery', 'Health', 'Sports',
    'Baby & Kids', 'Automotive'
]
cat_probs = np.random.dirichlet(np.ones(len(BASE_CATEGORIES)))  # non-uniform

CATEGORY_PROBS = dict(zip(BASE_CATEGORIES, cat_probs))

SUBCATEGORIES = {
    'Food & Beverages': {
        'Coffee': 0.22, 'Tea': 0.15, 'Spices': 0.12, 'Grains': 0.10,
        'Honey': 0.08, 'Oil': 0.08, 'Flour': 0.08, 'Sugar': 0.07, 'Beverages': 0.10
    },
    'Fashion': {
        'Traditional Wear': 0.28, 'Modern Clothing': 0.25, 'Shoes': 0.15,
        'Accessories': 0.12, 'Bags': 0.12, 'Jewelry': 0.08
    },
    'Beauty & Personal Care': {
        'Haircare': 0.24, 'Skincare': 0.22, 'Makeup': 0.20,
        'Fragrances': 0.18, 'Personal Hygiene': 0.16
    },
    'Electronics': {
        'Smartphones': 0.34, 'Laptops': 0.14, 'Televisions': 0.12,
        'Headphones': 0.10, 'Speakers': 0.09, 'Accessories': 0.08,
        'Cameras': 0.07, 'Watches': 0.06
    },
    'Home & Kitchen': {
        'Cookware': 0.20, 'Furniture': 0.20, 'Bedding': 0.15, 'Appliances': 0.14,
        'Cleaning': 0.12, 'Storage': 0.10, 'Lighting': 0.05, 'Decor': 0.04
    },
    'Books & Stationery': {
        'Fiction': 0.22, 'Educational': 0.30, 'Religion': 0.18,
        'Children': 0.18, 'Business': 0.12
    },
    'Health': {
        'Medicines': 0.30, 'Supplements': 0.25, 'First Aid': 0.20,
        'Medical Equipment': 0.25
    },
    'Sports': {
        'Equipment': 0.40, 'Sportswear': 0.25, 'Footwear': 0.20, 'Fitness': 0.15
    },
    'Baby & Kids': {
        'Clothing': 0.35, 'Toys': 0.25, 'Feeding': 0.20, 'Furniture': 0.20
    },
    'Automotive': {
        'Parts': 0.45, 'Accessories': 0.30, 'Tools': 0.15, 'Car Care': 0.10
    }
}

BRANDS = {
    'Electronics': [
        'TECNO', 'Infinix', 'Samsung', 'Hisense', 'Huawei', 'Dell', 'HP', 'Lenovo',
        'Apple', 'Xiaomi', 'Anker', 'Canon', 'Nikon', 'Asus', 'Acer', 'Transsion'
    ],
    'Food & Beverages': [
        'Ethio Coffee', 'Tomoca', "Kaldi's", 'Dashen', 'Awash', 'Sheba Honey',
        'Harar', 'Yirgacheffe', 'Green Farms', 'Limu Co.', 'Kaffa Select'
    ],
    'Fashion': [
        'Habesha Design', 'Sheba Crafts', 'Lucy Fashion', 'Tibeb Athletics',
        'Mekelle Leather', 'Gonder Textiles', 'Awraja', 'Addis Wear', 'Shiro Style'
    ],
    'Beauty & Personal Care': [
        'Ethio Herbal', 'Sheba Beauty', 'Nivea', 'Dove', 'Pantene', "L'Oréal",
        'Natural Touch', 'Natural Remedies', 'Addis Glow'
    ],
    'Home & Kitchen': [
        'Mega Mitad', 'Ethio Pottery', 'Addis Home', 'Sheger Furniture',
        'Blue Nile Crafts', 'Nile Decor', 'Tana Appliances', 'Abay Kitchen'
    ],
    'Books & Stationery': ['Unity Publishers', 'EPHI Press', 'Addis Books', 'Zemen Print', 'Ager Lit.'],
    'Health': ['Pharma Ethiopia', 'MediCare', 'Health Plus', 'EthioMed', 'Red Cross Supplies'],
    'Sports': ['Ethio Sports', "Runner's Choice", 'Tibeb Athletics', 'Addis Fit', 'Altitude Gear'],
    'Baby & Kids': ['Baby Ethiopia', 'Kids World', 'Happy Child', 'Addis Baby', 'MamaCare'],
    'Automotive': ['Auto Addis', 'Mekina Parts', 'Drive Safe', 'Ethio Motor', 'Blue Star Auto']
}

PHONE_MODELS = {
    'TECNO': ['Spark', 'Camon', 'Pova', 'Phantom'],
    'Infinix': ['Hot', 'Note', 'Zero', 'Smart'],
    'Samsung': ['Galaxy A', 'Galaxy M', 'Galaxy S'],
    'Huawei': ['P', 'Y', 'Nova'],
    'Xiaomi': ['Redmi', 'Mi', 'Poco']
}
LAPTOP_MODELS = {
    'Dell': ['Inspiron', 'Vostro', 'XPS'],
    'HP': ['Pavilion', 'EliteBook', 'ProBook'],
    'Lenovo': ['ThinkPad', 'IdeaPad'],
    'Asus': ['VivoBook', 'ZenBook'],
    'Acer': ['Aspire', 'Swift']
}
COFFEE_TYPES = ['Yirgacheffe', 'Sidamo', 'Harar', 'Limu', 'Jimma', 'Kaffa']

# Price ranges (ETB)
PRICE_RANGES = {
    'Electronics': {
        'Smartphones': (2500, 65_000), 'Laptops': (14_000, 140_000),
        'Televisions': (4_000, 120_000), 'Headphones': (400, 20_000),
        'Speakers': (700, 35_000), 'Accessories': (200, 8_000),
        'Cameras': (8_000, 180_000), 'Watches': (1_500, 60_000)
    },
    'Food & Beverages': {
        'Coffee': (150, 8_000), 'Tea': (50, 2_500), 'Spices': (80, 5_000),
        'Grains': (80, 3_000), 'Honey': (200, 5_500), 'Oil': (250, 6_000),
        'Sugar': (80, 3_000), 'Flour': (80, 3_000), 'Beverages': (200, 6_000)
    },
    'Fashion': {
        'Traditional Wear': (700, 28_000), 'Modern Clothing': (400, 18_000),
        'Shoes': (500, 12_000), 'Accessories': (150, 6_000), 'Bags': (400, 9_000), 'Jewelry': (600, 25_000)
    },
    'Beauty & Personal Care': {
        'Haircare': (150, 4_000), 'Skincare': (200, 5_000), 'Makeup': (300, 7_000), 'Fragrances': (500, 12_000),
        'Personal Hygiene': (150, 3_000)
    },
    'Home & Kitchen': {
        'Cookware': (300, 8_000), 'Furniture': (2_500, 80_000), 'Bedding': (800, 12_000),
        'Appliances': (2_000, 60_000), 'Cleaning': (150, 4_500), 'Storage': (250, 6_000),
        'Lighting': (300, 10_000), 'Decor': (400, 12_000)
    },
    'Books & Stationery': {'Fiction': (150, 2_500), 'Educational': (200, 5_000), 'Religion': (100, 2_000), 'Children': (120, 2_000), 'Business': (250, 4_500)},
    'Health': {'Medicines': (200, 7_000), 'Supplements': (150, 5_000), 'First Aid': (200, 3_500), 'Medical Equipment': (1_000, 35_000)},
    'Sports': {'Equipment': (800, 18_000), 'Sportswear': (300, 8_000), 'Footwear': (500, 10_000), 'Fitness': (600, 15_000)},
    'Baby & Kids': {'Clothing': (200, 6_000), 'Toys': (250, 8_000), 'Feeding': (200, 6_000), 'Furniture': (1_000, 20_000)},
    'Automotive': {'Parts': (800, 25_000), 'Accessories': (300, 8_000), 'Tools': (500, 15_000), 'Car Care': (300, 9_000)}
}

# Holidays (seasonal influence without adding columns)
HOLIDAYS = [
    'Enkutatash', 'Meskel', 'Timket', 'Genna', 'Fasika', 'Eid al-Fitr', 'Eid al-Adha'
]
HOLIDAY_TAGS = {
    'Fashion': ['Holiday Collection', 'Enkutatash Special', 'Timket White'],
    'Food & Beverages': ['Holiday Roast', 'Fasika Special', 'Genna Blend'],
    'Electronics': ['Holiday Offer', 'New Year Sale'],
    'Home & Kitchen': ['Holiday Set', 'Family Bundle'],
    'Beauty & Personal Care': ['Festive Glow', 'Holiday Fragrance']
}

# ------------------------------
# Helpers
# ------------------------------
def choose_with_probs(mapping):
    items = list(mapping.keys())
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return np.random.choice(items, p=probs)

def zipf_choice(items, alpha=1.4):
    # Zipf popularity to avoid uniform brand pick
    ranks = np.arange(1, len(items) + 1)
    weights = 1 / (ranks ** alpha)
    weights = weights / weights.sum()
    return np.random.choice(items, p=weights)

def jitter_text(text):
    # Randomly inject minor noise: casing, hyphens, extra tokens, typos
    variants = [
        text,
        text.lower(),
        text.upper(),
        text.replace(' ', '-'),
        text + ' ' + fake.word(),
        text + ' ' + fake.word() + ' ' + fake.word()
    ]
    # occasional tiny typo
    if len(text) > 6 and random.random() < 0.10:
        pos = random.randint(0, len(text) - 2)
        text = text[:pos] + text[pos+1] + text[pos] + text[pos+2:]
        variants.append(text)
    return random.choice(variants)

def generate_product_name(category, subcategory, brand):
    if category == 'Electronics':
        if subcategory == 'Smartphones':
            model = random.choice(PHONE_MODELS.get(brand, ['Smart']))
            number = random.choice(['10', '11', '12', '13', '20', '30', 'Pro', 'Lite', 'Max'])
            return jitter_text(f'{brand} {model} {number}')
        elif subcategory == 'Laptops':
            series = random.choice(LAPTOP_MODELS.get(brand, ['Series']))
            model_num = random.choice(['15', '14', '13', 'X360', 'G5'])
            return jitter_text(f'{brand} {series} {model_num}')
        elif subcategory == 'Televisions':
            sizes = ['32"', '40"', '43"', '50"', '55"', '65"']
            types = ['LED TV', 'Smart TV', 'UHD TV', 'Android TV']
            return jitter_text(f'{brand} {random.choice(sizes)} {random.choice(types)}')
    elif category == 'Food & Beverages':
        if subcategory == 'Coffee':
            coffee_type = random.choice(COFFEE_TYPES)
            weight = random.choice(['250g', '500g', '1kg'])
            return jitter_text(f'Ethiopian {coffee_type} Coffee {weight}')
        elif subcategory == 'Tea':
            types = ['Black', 'Green', 'Herbal', 'Traditional']
            weight = random.choice(['100g', '250g', '500g'])
            return jitter_text(f'{brand} {random.choice(types)} Tea {weight}')
    elif category == 'Fashion':
        if subcategory == 'Traditional Wear':
            items = ['Habesha Kemis', 'Kuta', 'Netela', 'Shawl', 'Dress']
            return jitter_text(f'{brand} {random.choice(items)}')
        elif subcategory == 'Modern Clothing':
            types = ['T-Shirt', 'Shirt', 'Dress', 'Pants', 'Jacket']
            return jitter_text(f'{brand} {random.choice(types)}')
    variants = ['Premium', 'Standard', 'Deluxe', 'Classic', 'Professional']
    return jitter_text(f'{brand} {subcategory} {random.choice(variants)}')

def generate_amharic_name(category, subcategory, brand):
    amharic_brands = {
        'TECNO': 'ቴክኖ', 'Infinix': 'ኢንፊኒክስ', 'Samsung': 'ሳምሰንግ', 'Hisense': 'ሃይሰንስ',
        'Dell': 'ዴል', 'HP': 'ኤችፒ', 'Lenovo': 'ሌኖቮ', 'Apple': 'አፕል',
        'Ethio Coffee': 'ኢትዮ ቡና', 'Sheba Crafts': 'ሸባ ስራ', 'Habesha Design': 'ሐበሻ ዲዛይን',
        'Tomoca': 'ቶሞካ', "Kaldi's": 'ካልዲስ', 'Dashen': 'ዳሸን', 'Awash': 'አዋሽ',
        'Harar': 'ሐረር', 'Yirgacheffe': 'ይርጋጨፍ'
    }
    amharic_sub = {
        'Smartphones': 'ስማርትፎን', 'Coffee': 'ቡና', 'Tea': 'ሻይ', 'Traditional Wear': 'ባህላዊ ልብስ',
        'Modern Clothing': 'ዘመናዊ ልብስ', 'Shoes': 'ጫማ', 'Accessories': 'ተጨማሪ ዕቃ',
        'Haircare': 'ፀጉር እንክብካቤ', 'Skincare': 'ቆዳ እንክብካቤ', 'Makeup': 'ማጅ',
        'Fragrances': 'ሽቶ', 'Laptops': 'ላፕቶፕ', 'Televisions': 'ቴሌቪዥን',
        'Headphones': 'ሄድፎን', 'Speakers': 'ስፒከር', 'Cameras': 'ካሜራ', 'Watches': 'ሰዓት'
    }
    brand_am = amharic_brands.get(brand, brand)
    sub_am = amharic_sub.get(subcategory, subcategory)
    # occasional suffix to reduce exact duplicates
    suffix = random.choice(['', ' ከፍተኛ ጥራት', ' ባህላዊ', ' ዘመናዊ'])
    return f'{brand_am} {sub_am}{suffix}'

def generate_description(category, subcategory, brand):
    base = [
        f'{brand} {subcategory} with premium features.',
        f'High-performance {subcategory.lower()} from {brand}.',
        f'{brand} latest {subcategory.lower()} model.',
        f'Authentic Ethiopian {subcategory.lower()} sourced locally.',
        f'Traditional {subcategory.lower()} with rich craftsmanship.'
    ]
    extra = [fake.sentence(), fake.text(max_nb_chars=120)]
    # inject stopwords/random tokens
    tokens = ' '.join(fake.words(nb=random.randint(3, 8)))
    desc = f"{random.choice(base)} {random.choice(extra)} {tokens}"
    return jitter_text(desc)

def generate_amharic_description(category, subcategory):
    choices = {
        'Electronics': [
            'ዘመናዊ ቴክኖሎጂ እና ከፍተኛ አፈጻጸም ያለው ምርት።',
            'ረጅም ጊዜ ለመጠቀም የሚረቁ።'
        ],
        'Food & Beverages': [
            'ንጹህ እና ጥራት ያለው ኢትዮጵያዊ ምርት።',
            'ከአካባቢ አርሶ አደሮች የተሰበሰበ።'
        ],
        'Fashion': [
            'ባህላዊ ዲዛይን እና ከፍተኛ ጥራት።',
            'የክብር ሰአት ለመስራት ተስማሚ።'
        ]
    }
    base = random.choice(choices.get(category, ['ከፍተኛ ጥራት ያለው ምርት።']))
    suffix = random.choice(['', ' እንኳን በአዲስ ዓመት በሰላም', ' ለእረፍት ይሆናል', ' ለቤት ተስማሚ'])
    return base + suffix

def seasonal_multiplier(category):
    # Apply holiday multiplier by category to price and delivery probabilities
    holiday = np.random.choice(HOLIDAYS, p=np.random.dirichlet(np.ones(len(HOLIDAYS))))
    mult = 1.0
    if category == 'Food & Beverages' and holiday in ['Genna', 'Fasika', 'Eid al-Fitr', 'Eid al-Adha']:
        mult = np.random.uniform(0.9, 1.15)  # some items get pricier; some discounted
    elif category == 'Fashion' and holiday in ['Enkutatash', 'Timket', 'Meskel']:
        mult = np.random.uniform(0.85, 1.10)
    elif category == 'Electronics' and holiday in ['Enkutatash', 'Genna']:
        mult = np.random.uniform(0.80, 1.05)
    else:
        mult = np.random.uniform(0.95, 1.05)
    return mult, holiday

def generate_price(category, subcategory):
    # Variable pricing with seasonal multiplier and noise
    if category in PRICE_RANGES and subcategory in PRICE_RANGES[category]:
        lo, hi = PRICE_RANGES[category][subcategory]
        base = np.random.randint(lo, hi + 1)
        mult, _ = seasonal_multiplier(category)
        price = int(base * mult)
        # add mild log-normal noise
        price = int(price * np.random.lognormal(mean=0.0, sigma=0.15))
        # realistic rounding
        if price < 1000:
            price = round(price / 10) * 10
        elif price < 10_000:
            price = round(price / 100) * 100
        else:
            price = round(price / 500) * 500
        # cap within extended plausible bounds
        price = max(50, min(price, 180_000))
        return price
    return np.random.randint(300, 12_000)

def generate_rating(brand):
    # Brand-based mean with randomness and clipping
    premium = {'Apple', 'Samsung', 'Dell', 'HP', 'Lenovo', 'Ethio Coffee', 'Habesha Design'}
    mu = 4.0 if brand in premium else 3.6
    rating = np.random.normal(mu, 0.35)
    return float(round(max(2.4, min(5.0, rating)), 1))

def generate_stock(brand, category):
    # Stock status varies by category and holiday demand
    in_stock_bias = 0.72 if category in ['Electronics', 'Food & Beverages'] else 0.64
    r = np.random.rand()
    if r < in_stock_bias:
        status = 'In Stock'
        qty = np.random.randint(5, 600)
        if brand in {'TECNO', 'Infinix', 'Ethio Coffee', 'Nivea', 'Dove'}:
            qty = np.random.randint(20, 1200)
    elif r < in_stock_bias + 0.20:
        status = 'Low Stock'
        qty = np.random.randint(1, 15)
    else:
        status = 'Out of Stock'
        qty = 0
    return status, int(qty)

def generate_weight(category, subcategory):
    # Same ranges but add slight noise to avoid clustering
    def clip(a, lo, hi): return round(float(max(lo, min(hi, a))), 2)
    if category == 'Electronics':
        if subcategory == 'Smartphones':
            return clip(np.random.uniform(0.14, 0.26), 0.12, 0.30)
        elif subcategory == 'Laptops':
            return clip(np.random.uniform(1.1, 2.7), 0.8, 3.5)
        elif subcategory == 'Televisions':
            return clip(np.random.uniform(4.5, 26.0), 2.0, 35.0)
        elif subcategory == 'Headphones':
            return clip(np.random.uniform(0.09, 0.6), 0.05, 0.9)
        elif subcategory == 'Speakers':
            return clip(np.random.uniform(0.45, 5.5), 0.2, 8.0)
    elif category == 'Food & Beverages':
        if subcategory in ['Coffee', 'Tea', 'Spices']:
            return clip(np.random.uniform(0.09, 1.05), 0.05, 2.0)
        elif subcategory in ['Grains', 'Flour', 'Sugar']:
            return clip(np.random.uniform(0.45, 5.2), 0.3, 8.0)
        elif subcategory == 'Honey':
            return clip(np.random.uniform(0.28, 2.1), 0.2, 3.0)
        elif subcategory == 'Oil':
            return clip(np.random.uniform(0.45, 3.2), 0.3, 4.0)
    elif category == 'Fashion':
        if subcategory == 'Shoes':
            return clip(np.random.uniform(0.45, 1.6), 0.2, 2.5)
        elif subcategory == 'Traditional Wear':
            return clip(np.random.uniform(0.28, 1.3), 0.2, 2.2)
        elif subcategory == 'Modern Clothing':
            return clip(np.random.uniform(0.18, 0.9), 0.1, 1.5)
        elif subcategory == 'Bags':
            return clip(np.random.uniform(0.28, 2.2), 0.2, 3.5)
    elif category == 'Beauty & Personal Care':
        if subcategory in ['Haircare', 'Skincare', 'Makeup']:
            return clip(np.random.uniform(0.08, 0.55), 0.05, 1.0)
        elif subcategory == 'Fragrances':
            return clip(np.random.uniform(0.04, 0.22), 0.02, 0.5)
    return clip(np.random.uniform(0.1, 5.0), 0.05, 10.0)

def generate_dimensions(category, subcategory):
    # Add random jitter to avoid identical sizes
    def dims(a, b, c, unit='cm'):
        return f"{int(a)}x{int(b)}x{int(c)} {unit}"
    if category == 'Electronics':
        if subcategory == 'Smartphones':
            return dims(np.random.randint(140, 166), np.random.randint(65, 81), np.random.randint(7, 11))
        elif subcategory == 'Laptops':
            return dims(np.random.randint(30, 41), np.random.randint(20, 26), np.random.randint(1, 4))
        elif subcategory == 'Televisions':
            return dims(np.random.randint(70, 151), np.random.randint(40, 91), np.random.randint(5, 16))
        elif subcategory == 'Headphones':
            return dims(np.random.randint(15, 26), np.random.randint(15, 21), np.random.randint(5, 11))
        elif subcategory == 'Speakers':
            return dims(np.random.randint(10, 41), np.random.randint(10, 31), np.random.randint(10, 26))
    elif category == 'Food & Beverages':
        if subcategory in ['Coffee', 'Tea', 'Spices']:
            return dims(np.random.randint(10, 21), np.random.randint(10, 16), np.random.randint(3, 9))
        elif subcategory in ['Grains', 'Flour', 'Sugar']:
            return dims(np.random.randint(20, 41), np.random.randint(15, 26), np.random.randint(5, 11))
        elif subcategory == 'Oil':
            return dims(np.random.randint(8, 16), np.random.randint(8, 16), np.random.randint(20, 31))
    elif category == 'Fashion':
        if subcategory == 'Shoes':
            return dims(np.random.randint(25, 36), np.random.randint(10, 16), np.random.randint(10, 16))
        elif subcategory in ['Traditional Wear', 'Modern Clothing']:
            return dims(np.random.randint(40, 61), np.random.randint(30, 51), np.random.randint(2, 6))
        elif subcategory == 'Bags':
            return dims(np.random.randint(25, 51), np.random.randint(15, 31), np.random.randint(5, 16))
    elif category == 'Beauty & Personal Care':
        if subcategory in ['Haircare', 'Skincare', 'Makeup']:
            return dims(np.random.randint(5, 16), np.random.randint(5, 11), np.random.randint(5, 11))
        elif subcategory == 'Fragrances':
            return dims(np.random.randint(8, 13), np.random.randint(3, 7), np.random.randint(3, 7))
    return dims(np.random.randint(10, 51), np.random.randint(5, 31), np.random.randint(2, 21))

def generate_warranty(category):
    if category == 'Electronics':
        return int(np.random.choice([6, 12, 24], p=[0.35, 0.45, 0.20]))
    return int(np.random.choice([0, 3, 6], p=[0.55, 0.35, 0.10]))

def generate_seller_name(category, brand):
    suffixes = {
        'Electronics': ['Tech Hub', 'Electronics Store', 'Digital Mart', 'Gadget Center', 'Byte Bazaar'],
        'Food & Beverages': ['Store', 'Market', 'Supplier', 'Distributor', 'Farmers Coop'],
        'Fashion': ['Boutique', 'Fashion House', 'Design Studio', 'Collection', 'Tailor'],
        'Beauty & Personal Care': ['Beauty Shop', 'Cosmetics Store', 'Care Center', 'Spa']
    }
    suffix = random.choice(suffixes.get(category, ['Store', 'Shop', 'Supplier', 'Mart']))
    # occasional transliteration noise
    b = brand if random.random() > 0.05 else brand.replace('a', 'aa').replace('e', 'ee')
    return f'{b} {suffix}'

def generate_seller_rating(product_rating):
    noise = np.random.normal(0.0, 0.25)
    sr = max(2.5, min(5.0, product_rating + noise))
    return float(round(sr, 1))

def delivery_probability(location):
    major = {'Addis Ababa', 'Bahir Dar', 'Hawassa', 'Mekelle', 'Dire Dawa'}
    base_yes = 0.83 if location in major else 0.70
    # holiday strain: sometimes lower probability
    strain = np.random.uniform(-0.08, 0.05)
    return max(0.55, min(0.95, base_yes + strain))

def generate_tags(category, subcategory, location, brand):
    tags = [category, subcategory, location, brand, 'Ethiopian-made']
    # Premium or campaign tags
    if brand in {'Apple', 'Samsung', 'Dell', 'Ethio Coffee', 'Habesha Design'} or random.random() < 0.22:
        tags.append('Premium')
    if random.random() < 0.28:
        tags.append('Best-Seller')
    if random.random() < 0.22:
        tags.append('New')
    # Holiday tags by category
    _, holiday = seasonal_multiplier(category)
    options = HOLIDAY_TAGS.get(category, [])
    if options and random.random() < 0.35:
        tags.append(random.choice(options))
    if random.random() < 0.18:
        tags.append(holiday)
    # occasional lowercase/comma noise
    tags = [t if random.random() > 0.15 else t.lower() for t in tags]
    return ','.join(tags)

# ------------------------------
# Generation
# ------------------------------
def generate_products(num_products=NUM_PRODUCTS):
    print(f"Generating {num_products:,} Ethiopian products (less predictable)...")
    products = []

    # Precompute location roulette
    loc_items = list(LOCATIONS.keys())
    loc_probs = np.array(list(LOCATIONS.values()), dtype=float)
    loc_probs = loc_probs / loc_probs.sum()

    # For each product
    for i in range(num_products):
        if i % 5000 == 0:
            print(f"Generated {i:,}...")
            sys.stdout.flush()

        product_id = f'ETP{i+1:05d}'

        # Category via drifting probability
        category = np.random.choice(BASE_CATEGORIES, p=list(CATEGORY_PROBS.values()))

        # Subcategory via local mapping (normalize)
        subs = SUBCATEGORIES.get(category, {'Standard': 1.0})
        sub_items = list(subs.keys())
        sub_probs = np.array(list(subs.values()), dtype=float)
        sub_probs = sub_probs / sub_probs.sum()
        subcategory = np.random.choice(sub_items, p=sub_probs)

        # Brand via Zipf popularity
        brand_list = BRANDS.get(category, ['Ethio Brand'])
        brand = zipf_choice(brand_list, alpha=np.random.uniform(1.2, 1.8))

        # Location
        location = np.random.choice(loc_items, p=loc_probs)

        # Product name & descriptions
        name = generate_product_name(category, subcategory, brand)
        name_amharic = generate_amharic_name(category, subcategory, brand)
        description = generate_description(category, subcategory, brand)
        description_amharic = generate_amharic_description(category, subcategory)

        # Price with seasonality and noise
        price = generate_price(category, subcategory)

        # Delivery availability
        yes_prob = delivery_probability(location)
        delivery_available = 'Yes' if np.random.rand() < yes_prob else 'No'

        # Rating, stock
        rating = generate_rating(brand)
        stock_status, stock_quantity = generate_stock(brand, category)

        # Weight, dimensions, warranty
        weight_kg = generate_weight(category, subcategory)
        dimensions = generate_dimensions(category, subcategory)
        warranty_months = generate_warranty(category)

        # Seller
        seller_name = generate_seller_name(category, brand)
        seller_rating = generate_seller_rating(rating)

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
            'price': int(price),
            'currency': CURRENCY,
            'brand': brand,
            'location': location,
            'delivery_available': delivery_available,
            'rating': float(rating),
            'stock_status': stock_status,
            'stock_quantity': int(stock_quantity),
            'weight_kg': float(weight_kg),
            'dimensions': dimensions,
            'warranty_months': int(warranty_months),
            'seller_name': seller_name,
            'seller_rating': float(seller_rating),
            'tags': tags
        }
        products.append(product)

    return pd.DataFrame(products, columns=COLUMNS)

def save_dataset(df, filename='ethiopian_products_150k.csv'):
    print(f"\nSaving {len(df):,} products to {filename}...")
    df.to_csv(filename, index=False, encoding='utf-8')
    sample_size = min(500, len(df))
    df.sample(sample_size, random_state=SEED).to_csv('ethiopian_products_sample.csv', index=False, encoding='utf-8')
    print(f"Complete dataset saved to {filename}")
    print(f"Sample of {sample_size} products saved to ethiopian_products_sample.csv")

def print_stats(df):
    print("\nDATASET STATISTICS")
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

    print("\nCATEGORY DISTRIBUTION")
    print("-" * 40)
    cat_counts = df['category'].value_counts()
    for category, count in cat_counts.items():
        pct = (count / len(df)) * 100
        avg_price = df[df['category'] == category]['price'].mean()
        print(f"{category:25} {count:7,} ({pct:5.1f}%) | Avg Price: {avg_price:,.0f} ETB")

    print("\nTOP 15 BRANDS")
    print("-" * 40)
    top_brands = df['brand'].value_counts().head(15)
    for brand, count in top_brands.items():
        print(f"{brand:20} {count:7,}")

if __name__ == "__main__":
    print("=" * 60)
    print("REALISTIC ETHIOPIAN E-COMMERCE PRODUCT DATASET GENERATOR (150k)")
    print("=" * 60)
    df = generate_products(NUM_PRODUCTS)
    save_dataset(df, 'ethiopian_products_150k.csv')
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print_stats(df)