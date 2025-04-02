import re
import csv
import time
import random
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- String Similarity Functions ---

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def jaro_similarity(s1, s2):
    """
    Calculate Jaro similarity between two strings.
    """
    # If the strings are identical, return 1.0
    if s1 == s2:
        return 1.0

    # If either string is empty, return 0.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    # Calculate match distance
    match_distance = max(len(s1), len(s2)) // 2 - 1
    match_distance = max(0, match_distance)

    # Count matching characters
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)

    matches = 0
    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    transpositions = 0
    k = 0

    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while k < len(s2) and not s2_matches[k]:
            k += 1
        if k < len(s2) and s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions = transpositions // 2

    # Calculate Jaro similarity
    return (matches / len(s1) + matches / len(s2) + (matches - transpositions) / matches) / 3.0

def jaro_winkler_similarity(s1, s2, p=0.1):
    """
    Calculate Jaro-Winkler similarity between two strings.
    """
    jaro_sim = jaro_similarity(s1, s2)
    
    # Calculate common prefix length (up to 4 characters)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    # Calculate Jaro-Winkler similarity
    return jaro_sim + (prefix_len * p * (1 - jaro_sim))

def tokenize_address(address):
    """
    Split address into tokens, useful for n-gram based matching.
    """
    return re.findall(r'\b\w+\b', address.lower())

def jaccard_similarity(s1, s2):
    """
    Calculate Jaccard similarity between two strings.
    """
    tokens1 = set(tokenize_address(s1))
    tokens2 = set(tokenize_address(s2))
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def get_n_grams(tokens, n=2):
    """
    Generate n-grams from a list of tokens.
    """
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def cosine_similarity_ngrams(addr1, addr2, n=2):
    """
    Calculate cosine similarity using n-grams.
    """
    tokens1 = tokenize_address(addr1)
    tokens2 = tokenize_address(addr2)
    
    # If tokens are too short for n-grams, reduce n
    if len(tokens1) < n or len(tokens2) < n:
        n = min(len(tokens1), len(tokens2), n)
        if n < 1:
            return 0.0
    
    # Get n-grams
    ngrams1 = set(get_n_grams(tokens1, n))
    ngrams2 = set(get_n_grams(tokens2, n))
    
    # Calculate intersection and union
    intersection = ngrams1.intersection(ngrams2)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    # Calculate cosine similarity
    return len(intersection) / (len(ngrams1) * len(ngrams2))**0.5

# --- Address Standardization ---

class AddressStandardizer:
    def __init__(self):
        """
        Initialize the address standardizer with abbreviation mappings.
        """
        # Street type abbreviations
        self.street_types = {
            'avenue': 'ave',
            'boulevard': 'blvd',
            'circle': 'cir',
            'court': 'ct',
            'drive': 'dr',
            'expressway': 'expy',
            'freeway': 'fwy',
            'highway': 'hwy',
            'lane': 'ln',
            'parkway': 'pkwy',
            'place': 'pl',
            'plaza': 'plz',
            'road': 'rd',
            'square': 'sq',
            'street': 'st',
            'terrace': 'ter',
            'trail': 'trl',
            'turnpike': 'tpke',
            'way': 'way'
        }
        
        # Unit type abbreviations
        self.unit_types = {
            'apartment': 'apt',
            'building': 'bldg',
            'department': 'dept',
            'floor': 'fl',
            'room': 'rm',
            'suite': 'ste',
            'unit': 'unit'
        }
        
        # Directional abbreviations
        self.directionals = {
            'north': 'n',
            'south': 's',
            'east': 'e',
            'west': 'w',
            'northeast': 'ne',
            'northwest': 'nw',
            'southeast': 'se',
            'southwest': 'sw'
        }
        
        # State abbreviations
        self.states = {
            'alabama': 'al',
            'alaska': 'ak',
            'arizona': 'az',
            'arkansas': 'ar',
            'california': 'ca',
            'colorado': 'co',
            'connecticut': 'ct',
            'delaware': 'de',
            'florida': 'fl',
            'georgia': 'ga',
            'hawaii': 'hi',
            'idaho': 'id',
            'illinois': 'il',
            'indiana': 'in',
            'iowa': 'ia',
            'kansas': 'ks',
            'kentucky': 'ky',
            'louisiana': 'la',
            'maine': 'me',
            'maryland': 'md',
            'massachusetts': 'ma',
            'michigan': 'mi',
            'minnesota': 'mn',
            'mississippi': 'ms',
            'missouri': 'mo',
            'montana': 'mt',
            'nebraska': 'ne',
            'nevada': 'nv',
            'new hampshire': 'nh',
            'new jersey': 'nj',
            'new mexico': 'nm',
            'new york': 'ny',
            'north carolina': 'nc',
            'north dakota': 'nd',
            'ohio': 'oh',
            'oklahoma': 'ok',
            'oregon': 'or',
            'pennsylvania': 'pa',
            'rhode island': 'ri',
            'south carolina': 'sc',
            'south dakota': 'sd',
            'tennessee': 'tn',
            'texas': 'tx',
            'utah': 'ut',
            'vermont': 'vt',
            'virginia': 'va',
            'washington': 'wa',
            'west virginia': 'wv',
            'wisconsin': 'wi',
            'wyoming': 'wy'
        }
        
        # Compile regex patterns for address components
        self.house_number_pattern = re.compile(r'^\d+')
        self.zip_pattern = re.compile(r'\b\d{5}(?:-\d{4})?\b')
        self.unit_pattern = re.compile(r'\b(?:apt|suite|ste|unit|bldg|floor|fl|#)\s*(?:[a-zA-Z0-9-]+)\b', re.IGNORECASE)
    
    def standardize(self, address):
        """
        Standardize an address by applying normalization rules.
        """
        if not address:
            return ""
            
        # Convert to lowercase
        addr = address.lower()
        
        # Replace periods and commas with spaces
        addr = addr.replace('.', ' ').replace(',', ' ')
        
        # Replace hash/pound/number sign with 'unit'
        addr = re.sub(r'#\s*', 'unit ', addr)
        
        # Standardize street types
        for full, abbr in self.street_types.items():
            # Replace full word with abbreviation
            addr = re.sub(r'\b' + full + r'\b', abbr, addr)
            # Also handle with period
            addr = re.sub(r'\b' + abbr + r'\.\b', abbr, addr)
        
        # Standardize unit types
        for full, abbr in self.unit_types.items():
            addr = re.sub(r'\b' + full + r'\b', abbr, addr)
            addr = re.sub(r'\b' + abbr + r'\.\b', abbr, addr)
        
        # Standardize directionals
        for full, abbr in self.directionals.items():
            addr = re.sub(r'\b' + full + r'\b', abbr, addr)
            addr = re.sub(r'\b' + abbr + r'\.\b', abbr, addr)
        
        # Standardize state names
        for full, abbr in self.states.items():
            addr = re.sub(r'\b' + full + r'\b', abbr, addr)
        
        # Remove extra whitespace
        addr = ' '.join(addr.split())
        
        return addr
    
    def parse_components(self, address):
        """
        Extract components from an address.
        """
        components = {
            'house_number': None,
            'street_name': None,
            'unit': None,
            'city': None,
            'state': None,
            'zipcode': None
        }
        
        # Extract ZIP code
        zip_match = self.zip_pattern.search(address)
        if zip_match:
            components['zipcode'] = zip_match.group()
            # Remove ZIP from address for further processing
            address = address.replace(components['zipcode'], '')
        
        # Extract unit information
        unit_match = self.unit_pattern.search(address)
        if unit_match:
            components['unit'] = unit_match.group()
            # Remove unit from address for further processing
            address = address.replace(components['unit'], '')
        
        # Extract house number (assuming it's at the beginning)
        house_match = self.house_number_pattern.search(address)
        if house_match:
            components['house_number'] = house_match.group()
        
        # Clean up remaining whitespace
        address = ' '.join(address.split())
        
        return components

# --- Feature Generation ---

def generate_features(addr1, addr2, standardizer):
    """
    Generate features for a pair of addresses.
    """
    features = {}
    
    # Standardize addresses
    std_addr1 = standardizer.standardize(addr1)
    std_addr2 = standardizer.standardize(addr2)
    
    # Calculate string similarity metrics
    features['levenshtein_sim'] = 1 - (levenshtein_distance(std_addr1, std_addr2) / max(len(std_addr1), len(std_addr2)) if max(len(std_addr1), len(std_addr2)) > 0 else 0)
    features['jaro_winkler_sim'] = jaro_winkler_similarity(std_addr1, std_addr2)
    features['jaccard_sim'] = jaccard_similarity(std_addr1, std_addr2)
    features['cosine_sim'] = cosine_similarity_ngrams(std_addr1, std_addr2)
    features['seq_matcher_sim'] = SequenceMatcher(None, std_addr1, std_addr2).ratio()
    
    # Extract address components
    comp1 = standardizer.parse_components(std_addr1)
    comp2 = standardizer.parse_components(std_addr2)
    
    # House number features
    if comp1['house_number'] and comp2['house_number']:
        features['house_num_match'] = int(comp1['house_number'] == comp2['house_number'])
        features['house_num_sim'] = 1 - (levenshtein_distance(comp1['house_number'], comp2['house_number']) / max(len(comp1['house_number']), len(comp2['house_number'])) if max(len(comp1['house_number']), len(comp2['house_number'])) > 0 else 0)
    else:
        features['house_num_match'] = 0
        features['house_num_sim'] = 0
    
    # Unit features
    if comp1['unit'] and comp2['unit']:
        features['unit_match'] = int(comp1['unit'] == comp2['unit'])
        features['unit_sim'] = jaro_winkler_similarity(comp1['unit'], comp2['unit'])
    else:
        features['unit_match'] = 0
        features['unit_sim'] = 0
    
    # Zipcode features
    if comp1['zipcode'] and comp2['zipcode']:
        features['zipcode_match'] = int(comp1['zipcode'] == comp2['zipcode'])
        features['zipcode_sim'] = 1 - (levenshtein_distance(comp1['zipcode'], comp2['zipcode']) / 5)  # Assuming 5-digit ZIP
    else:
        features['zipcode_match'] = 0
        features['zipcode_sim'] = 0
    
    # Length features
    features['len_diff'] = abs(len(std_addr1) - len(std_addr2)) / max(len(std_addr1), len(std_addr2)) if max(len(std_addr1), len(std_addr2)) > 0 else 0
    features['token_count_diff'] = abs(len(tokenize_address(std_addr1)) - len(tokenize_address(std_addr2))) / max(len(tokenize_address(std_addr1)), len(tokenize_address(std_addr2))) if max(len(tokenize_address(std_addr1)), len(tokenize_address(std_addr2))) > 0 else 0
    
    return features

# --- Data Generation and Model Training ---

def generate_sample_data(n_samples=1000, n_variations=3):
    """
    Generate sample address data with variations for training.
    """
    # Base addresses (real-world examples)
    base_addresses = [
        {'id': 'A1', 'address': '123 Main Street, Apt 4B, New York, NY 10001'},
        {'id': 'A2', 'address': '456 Oak Avenue, Suite 202, Los Angeles, CA 90001'},
        {'id': 'A3', 'address': '789 Pine Road, Chicago, IL 60601'},
        {'id': 'A4', 'address': '321 Cedar Boulevard, San Francisco, CA 94101'},
        {'id': 'A5', 'address': '555 Maple Lane, Boston, MA 02101'},
        {'id': 'A6', 'address': '987 Elm Street, Unit 3C, Miami, FL 33101'},
        {'id': 'A7', 'address': '654 Birch Drive, Seattle, WA 98101'},
        {'id': 'A8', 'address': '234 Walnut Court, Apt 7D, Philadelphia, PA 19101'},
        {'id': 'A9', 'address': '876 Cherry Place, Denver, CO 80201'},
        {'id': 'A10', 'address': '432 Spruce Way, Apartment 5A, Atlanta, GA 30301'},
        {'id': 'A11', 'address': '111 River Road, Unit 2B, Nashville, TN 37201'},
        {'id': 'A12', 'address': '222 Lake Drive, Portland, OR 97201'},
        {'id': 'A13', 'address': '333 Mountain Avenue, Suite 101, Phoenix, AZ 85001'},
        {'id': 'A14', 'address': '444 Forest Lane, Apt 9C, Austin, TX 78701'},
        {'id': 'A15', 'address': '555 Ocean Boulevard, San Diego, CA 92101'}
    ]
    
    # Helper function to create address variations
    def create_variation(address, variation_level=1):
        # Different types of variations based on the level
        addr = address['address']
        
        if variation_level == 1:
            # Minor variations (abbreviations, formatting)
            variations = [
                addr.replace('Street', 'St').replace('Avenue', 'Ave').replace('Road', 'Rd'),
                addr.replace('Boulevard', 'Blvd').replace('Lane', 'Ln').replace('Drive', 'Dr'),
                addr.replace('Apartment', 'Apt').replace('Suite', 'Ste').replace('Unit', '#'),
                addr.replace(', ', ' ').replace(', ', ' '),
                addr.replace('New York', 'NY').replace('Los Angeles', 'LA').replace('San Francisco', 'SF')
            ]
        elif variation_level == 2:
            # Moderate variations (typos, omissions)
            typos = {
                'Street': 'Sreet', 'Avenue': 'Avene', 'Road': 'Raod',
                'Boulevard': 'Bouelvard', 'Lane': 'Lne', 'Drive': 'Drve',
                'New York': 'New Yrok', 'Los Angeles': 'Los Angles', 'Chicago': 'Chcago'
            }
            
            var1 = addr
            # Apply 1-2 random typos
            for _ in range(random.randint(1, 2)):
                key = random.choice(list(typos.keys()))
                if key in var1:
                    var1 = var1.replace(key, typos[key])
            
            # Omit parts
            var2 = re.sub(r', [A-Z]{2} \d{5}', '', addr)  # Remove state and ZIP
            var3 = re.sub(r'Apt \w+,|Suite \w+,|Unit \w+,|#\w+,', '', addr)  # Remove unit
            var4 = addr.split(',')[0]  # Just street address
            
            variations = [var1, var2, var3, var4]
        else:
            # Major variations (different formats, word order changes)
            components = addr.split(',')
            if len(components) >= 3:
                var1 = f"{components[1].strip()}, {components[0].strip()}, {components[2].strip()}"  # Reordered
            else:
                var1 = addr
            
            # Different format
            house_num = re.search(r'^\d+', addr)
            street = re.search(r'\d+\s+(.+?)(?:,|$)', addr)
            if house_num and street:
                var2 = addr.replace(f"{house_num.group()} {street.group(1)}", f"{street.group(1)} {house_num.group()}")
            else:
                var2 = addr
            
            # Missing parts and different separators
            var3 = addr.replace(', ', ' - ').replace(', ', ' - ')
            var4 = re.sub(r'\b(Apt|Suite|Unit|#)\b.*?,', '', addr)  # Remove unit designator and number
            
            variations = [var1, var2, var3, var4]
        
        # Select a random variation
        variation = random.choice(variations)
        return {'id': f"V{address['id'][1:]}-{variation_level}", 'address': variation}
    
    # Generate positive pairs (true matches with variations)
    positive_pairs = []
    for base_addr in base_addresses:
        for level in range(1, n_variations + 1):
            var_addr = create_variation(base_addr, level)
            positive_pairs.append({
                'addr1': base_addr['address'],
                'addr2': var_addr['address'],
                'id1': base_addr['id'],
                'id2': var_addr['id'],
                'is_match': 1
            })
    
    # Generate negative pairs (non-matches)
    negative_pairs = []
    for i in range(len(base_addresses)):
        for j in range(i + 1, len(base_addresses)):
            negative_pairs.append({
                'addr1': base_addresses[i]['address'],
                'addr2': base_addresses[j]['address'],
                'id1': base_addresses[i]['id'],
                'id2': base_addresses[j]['id'],
                'is_match': 0
            })
    
    # Create additional negative pairs with variations
    additional_negative = []
    base_ids = [addr['id'] for addr in base_addresses]
    for pair in positive_pairs:
        # Find a base address that's not the match for this variation
        other_bases = [addr for addr in base_addresses if addr['id'] != pair['id1']]
        if other_bases:
            other_base = random.choice(other_bases)
            additional_negative.append({
                'addr1': other_base['address'],
                'addr2': pair['addr2'],
                'id1': other_base['id'],
                'id2': pair['id2'],
                'is_match': 0
            })
    
    # Combine positive and negative pairs
    all_pairs = positive_pairs + negative_pairs + additional_negative
    
    # Shuffle and limit to requested sample size
    random.shuffle(all_pairs)
    return all_pairs[:n_samples]

def train_xgboost_model(data, standardizer):
    """
    Train an XGBoost model for address matching.
    """
    print("Generating features for training data...")
    start_time = time.time()
    
    # Generate features for each address pair
    features_list = []
    labels = []
    
    for i, pair in enumerate(data):
        features = generate_features(pair['addr1'], pair['addr2'], standardizer)
        features_list.append(features)
        labels.append(pair['is_match'])
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} pairs")
    
    # Convert to DataFrame for easier handling
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    print(f"Feature generation completed in {time.time() - start_time:.2f} seconds")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost model...")
    
    # Create and train the model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 important features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    return model, X.columns

def predict_matches(model, feature_columns, addr_pairs, standardizer, threshold=0.5):
    """
    Predict matches for new address pairs.
    """
    predictions = []
    
    for pair in addr_pairs:
        # Generate features
        features = generate_features(pair['addr1'], pair['addr2'], standardizer)
        
        # Convert to DataFrame with the same columns as training data
        features_df = pd.DataFrame([features], columns=feature_columns)
        
        # Fill any missing columns with 0
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Ensure columns are in the same order
        features_df = features_df[feature_columns]
        
        # Predict probability
        prob = model.predict_proba(features_df)[0][1]
        
        # Add prediction to results
        predictions.append({
            'addr1': pair['addr1'],
            'addr2': pair['addr2'],
            'match_probability': prob,
            'predicted_match': prob >= threshold
        })
    
    return predictions

# --- Main Execution ---

def main():
    # Initialize address standardizer
    standardizer = AddressStandardizer()
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_sample_data(n_samples=1000, n_variations=3)
    
    # Train XGBoost model
    model, feature_columns = train_xgboost_model(training_data, standardizer)
    
    # Example test data
    test_data = [
        {
            'addr1': '123 Main Street, Apt 4B, New York, NY 10001',
            'addr2': '123 Main St, #4B, New York, NY 10001'
        },
        {
            'addr1': '456 Oak Avenue, Suite 202, Los Angeles, CA 90001',
            'addr2': '456 Oak Ave. Suite 202, Los Angeles, California 90001'
        },
        {
            'addr1': '789 Pine Road, Chicago, IL 60601',
            'addr2': '789 Pine Rd, Chicago, Illinois 60601'
        },
        {
            'addr1': '123 Main Street, Apt 4B, New York, NY 10001',
            'addr2': '789 Pine Road, Chicago, IL 60601'
        },
        {
            'addr1': '555 Maple Lane, Boston, MA 02101',
            'addr2': '555 Maple Lane, unit 3C, Boston, MA 02101'
        }
    ]
    
    # Predict matches for test data
    print("\nTesting the model on example address pairs:")
    predictions = predict_matches(model, feature_columns, test_data, standardizer)
    
    # Display results
    print("\nMatch Predictions:")
    for pred in predictions:
        print(f"Address 1: {pred['addr1']}")
        print(f"Address 2: {pred['addr2']}")
        print(f"Match Probability: {pred['match_probability']:.4f}")
        print(f"Predicted Match: {'Yes' if pred['predicted_match'] else 'No'}")
        print("-" * 50)
    
    # Save model
    model.save_model('address_matching_model.json')
    print("Model saved to address_matching_model.json")

if __name__ == "__main__":
    main()