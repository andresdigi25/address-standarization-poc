import re
import csv
import time
from difflib import SequenceMatcher

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

def jaro_winkler_similarity(s1, s2, p=0.1):
    """
    Calculate Jaro-Winkler similarity between two strings.
    """
    # If the strings are identical, return 1.0
    if s1 == s2:
        return 1.0

    # If either string is empty, return 0.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    # Calculate Jaro similarity
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
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions = transpositions // 2

    # Calculate Jaro similarity
    jaro = (matches / len(s1) + matches / len(s2) + (matches - transpositions) / matches) / 3.0

    # Calculate common prefix length (up to 4 characters)
    prefix_len = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    # Calculate Jaro-Winkler similarity
    jaro_winkler = jaro + (prefix_len * p * (1 - jaro))

    return jaro_winkler

def tokenize_address(address):
    """
    Split address into tokens, useful for n-gram based matching.
    """
    return re.findall(r'\b\w+\b', address.lower())

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
    
    # Get n-grams
    ngrams1 = set(get_n_grams(tokens1, n))
    ngrams2 = set(get_n_grams(tokens2, n))
    
    # Calculate intersection and union
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

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

class AddressMatcher:
    def __init__(self):
        """
        Initialize the address matcher with standardizer and thresholds.
        """
        self.standardizer = AddressStandardizer()
        self.threshold = 0.8  # Default threshold for matching
        
        # Component weights
        self.weights = {
            'house_number': 0.35,
            'street_name': 0.25,
            'unit': 0.15,
            'zipcode': 0.25
        }
    
    def string_similarity(self, s1, s2):
        """
        Calculate string similarity using a combination of algorithms.
        """
        if not s1 or not s2:
            return 0.0
            
        # For short strings, use Levenshtein
        if len(s1) < 5 or len(s2) < 5:
            lev_sim = 1 - (levenshtein_distance(s1, s2) / max(len(s1), len(s2)))
            return lev_sim
            
        # For longer strings, use a combination
        jaro_sim = jaro_winkler_similarity(s1, s2)
        seq_sim = SequenceMatcher(None, s1, s2).ratio()
        
        # Use the higher similarity score with a bias toward Jaro-Winkler
        return jaro_sim * 0.7 + seq_sim * 0.3
    
    def compare_address_components(self, comp1, comp2):
        """
        Compare individual address components.
        """
        similarities = {}
        total_weight = 0
        
        for component, weight in self.weights.items():
            val1 = comp1.get(component)
            val2 = comp2.get(component)
            
            if val1 and val2:
                # For house numbers and ZIP codes, use exact matching
                if component in ['house_number', 'zipcode']:
                    similarities[component] = 1.0 if val1 == val2 else 0.0
                else:
                    similarities[component] = self.string_similarity(val1, val2)
                
                total_weight += weight
        
        # If we couldn't compare any components, return 0
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted similarity
        weighted_sim = sum(similarities.get(comp, 0) * weight 
                           for comp, weight in self.weights.items())
        
        return weighted_sim / total_weight
    
    def calculate_similarity(self, addr1, addr2):
        """
        Calculate overall similarity between two addresses.
        """
        # Standardize addresses
        std_addr1 = self.standardizer.standardize(addr1)
        std_addr2 = self.standardizer.standardize(addr2)
        
        # Calculate overall string similarity
        overall_sim = self.string_similarity(std_addr1, std_addr2)
        
        # Extract and compare components
        comp1 = self.standardizer.parse_components(std_addr1)
        comp2 = self.standardizer.parse_components(std_addr2)
        component_sim = self.compare_address_components(comp1, comp2)
        
        # N-gram similarity for catching transpositions and word order differences
        ngram_sim = cosine_similarity_ngrams(std_addr1, std_addr2)
        
        # Calculate final similarity score with weights
        final_sim = (overall_sim * 0.3) + (component_sim * 0.5) + (ngram_sim * 0.2)
        
        return final_sim
    
    def find_matches(self, addresses1, addresses2, threshold=None):
        """
        Find matches between two lists of addresses.
        """
        if threshold is None:
            threshold = self.threshold
            
        matches = []
        
        print(f"Comparing {len(addresses1)} addresses against {len(addresses2)} addresses...")
        start_time = time.time()
        
        for i, addr1 in enumerate(addresses1):
            best_match = None
            best_score = threshold  # Only consider matches above threshold
            
            # Print progress every 10 addresses
            if i > 0 and i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i} addresses ({i/len(addresses1)*100:.1f}%) in {elapsed:.2f} seconds")
            
            for addr2 in addresses2:
                similarity = self.calculate_similarity(addr1['address'], addr2['address'])
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = addr2
            
            if best_match:
                matches.append({
                    'address1': addr1,
                    'address2': best_match,
                    'similarity': best_score
                })
        
        total_time = time.time() - start_time
        print(f"Matching completed in {total_time:.2f} seconds. Found {len(matches)} matches.")
        
        return matches

# Sample dataset generator
def generate_sample_data():
    """
    Generate sample address datasets with variations.
    """
    original_addresses = [
        {'id': 'A1', 'address': '123 Main Street, Apt 4B, New York, NY 10001'},
        {'id': 'A2', 'address': '456 Oak Avenue, Suite 202, Los Angeles, CA 90001'},
        {'id': 'A3', 'address': '789 Pine Road, Chicago, IL 60601'},
        {'id': 'A4', 'address': '321 Cedar Boulevard, San Francisco, CA 94101'},
        {'id': 'A5', 'address': '555 Maple Lane, Boston, MA 02101'},
        {'id': 'A6', 'address': '987 Elm Street, Unit 3C, Miami, FL 33101'},
        {'id': 'A7', 'address': '654 Birch Drive, Seattle, WA 98101'},
        {'id': 'A8', 'address': '234 Walnut Court, Apt 7D, Philadelphia, PA 19101'},
        {'id': 'A9', 'address': '876 Cherry Place, Denver, CO 80201'},
        {'id': 'A10', 'address': '432 Spruce Way, Apartment 5A, Atlanta, GA 30301'}
    ]
    
    # Create variations for the second dataset
    variations = [
        {'id': 'B1', 'address': '123 Main St, #4B, New York, New York 10001'},
        {'id': 'B2', 'address': '456 Oak Ave. Ste 202, Los Angeles, California 90001'},
        {'id': 'B3', 'address': '789 Pine Rd, Chicago, Illinois 60601'},
        {'id': 'B4', 'address': '321 Cedar Blvd, SF, CA 94101'},
        {'id': 'B5', 'address': '555 Maple Ln, Apt 3C, Boston, MA 02101'},  # Unit number changed
        {'id': 'B6', 'address': '987 Elm St, Unit 3C, Miami, Florida 33101'},
        {'id': 'B7', 'address': '654 Birch Dr., Seattle WA 98101'},
        {'id': 'B8', 'address': '234 Walnut Ct Apartment 7D, Phila., PA 19101'},
        {'id': 'B9', 'address': '876 Cherry Pl., Denver Colorado 80201'},
        {'id': 'B10', 'address': '432 Spruce Wy, Apt. 5A, Atlanta GA 30301'},
        {'id': 'B11', 'address': '111 River Road, Nashville, TN 37201'},  # No match in dataset 1
        {'id': 'B12', 'address': '222 Lake Drive, Portland, OR 97201'}  # No match in dataset 1
    ]
    
    return original_addresses, variations

# Function to output matches to CSV
def output_matches_to_csv(matches, filename='address_matches.csv'):
    """
    Write matches to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['address1_id', 'address1', 'address2_id', 'address2', 'similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for match in matches:
            writer.writerow({
                'address1_id': match['address1']['id'],
                'address1': match['address1']['address'],
                'address2_id': match['address2']['id'],
                'address2': match['address2']['address'],
                'similarity': match['similarity']
            })
    
    print(f"Matches written to {filename}")

# Main function
def main():
    # Generate sample data
    dataset1, dataset2 = generate_sample_data()
    
    # Initialize the address matcher
    matcher = AddressMatcher()
    
    # Example of standardizing addresses
    print("Example of address standardization:")
    for i, address in enumerate(dataset1[:3]):
        original = address['address']
        standardized = matcher.standardizer.standardize(original)
        components = matcher.standardizer.parse_components(standardized)
        print(f"Original: {original}")
        print(f"Standardized: {standardized}")
        print(f"Components: {components}")
        print()
    
    # Find matches between datasets
    matches = matcher.find_matches(dataset1, dataset2, threshold=0.7)
    
    # Output matches
    print("\nTop matches found:")
    for match in sorted(matches, key=lambda x: x['similarity'], reverse=True):
        print(f"Similarity: {match['similarity']:.4f}")
        print(f"  Address 1: {match['address1']['id']} - {match['address1']['address']}")
        print(f"  Address 2: {match['address2']['id']} - {match['address2']['address']}")
        print()
    
    # Save matches to CSV
    output_matches_to_csv(matches)

if __name__ == "__main__":
    main()