import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from fuzzywuzzy import fuzz
import usaddress
import pickle
import os
import time
import math
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from typing import List, Dict, Tuple, Any, Set, Optional

class USAddressRecordLinkage:
    """
    Record linkage system for US healthcare addresses, connecting
    point of care (POC) parent records with inventory atoms data.
    """
    
    def __init__(self, model_path=None, smarty_auth_id=None, smarty_auth_token=None):
        """Initialize the record linkage system"""
        self.model = None
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        # Constants
        self.HIGH_THRESHOLD = 0.85  # Definite match threshold
        self.LOW_THRESHOLD = 0.40   # Possible match threshold
        
        # Geocoding services
        self.smarty_auth_id = smarty_auth_id
        self.smarty_auth_token = smarty_auth_token
        self.use_smartystreets = smarty_auth_id is not None and smarty_auth_token is not None
        
        # Set up Nominatim geocoder with appropriate user agent
        self.geocoder = Nominatim(user_agent="healthcare_address_linkage")
        
        # Cache for geocoding results to minimize API calls
        self.geocode_cache = {}
        
        # USPS standardization dictionaries
        self.street_suffix_map = {
            'AVENUE': 'AVE', 'BOULEVARD': 'BLVD', 'CIRCLE': 'CIR',
            'COURT': 'CT', 'DRIVE': 'DR', 'EXPRESSWAY': 'EXPY',
            'HIGHWAY': 'HWY', 'LANE': 'LN', 'PARKWAY': 'PKWY',
            'PLACE': 'PL', 'ROAD': 'RD', 'SQUARE': 'SQ',
            'STREET': 'ST', 'TERRACE': 'TER'
            # Add more as needed
        }
        
        self.directional_map = {
            'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W',
            'NORTHEAST': 'NE', 'NORTHWEST': 'NW', 'SOUTHEAST': 'SE', 
            'SOUTHWEST': 'SW'
        }
        
        self.unit_type_map = {
            'APARTMENT': 'APT', 'BUILDING': 'BLDG', 'DEPARTMENT': 'DEPT',
            'FLOOR': 'FL', 'ROOM': 'RM', 'SUITE': 'STE', 'UNIT': 'UNIT'
            # Add more as needed
        }
        
        self.healthcare_term_map = {
            'HOSPITAL': 'HOSP', 'MEDICAL CENTER': 'MED CTR', 'CLINIC': 'CLIN',
            'HEALTHCARE': 'HLTHCARE', 'HEALTH CARE': 'HLTHCARE',
            'SAINT': 'ST', 'MOUNT': 'MT', 'CENTER': 'CTR'
            # Add more as needed
        }

    # STEP 1: DATA PREPARATION AND STANDARDIZATION
    def standardize_address(self, record: Dict) -> Dict:
        """Standardize a US address record"""
        # Create a copy to avoid modifying the original
        std_record = record.copy()
        
        # Extract address components using usaddress
        address_str = record.get('address', '')
        if not address_str:
            return std_record
        
        try:
            parsed_addr, addr_type = usaddress.tag(address_str)
            
            # Standardize street name and number
            if 'AddressNumber' in parsed_addr:
                std_record['street_number'] = parsed_addr['AddressNumber']
            
            # Standardize street name
            street_name_parts = []
            if 'StreetNamePreDirectional' in parsed_addr:
                dir_raw = parsed_addr['StreetNamePreDirectional'].upper()
                street_name_parts.append(self.directional_map.get(dir_raw, dir_raw))
            
            if 'StreetName' in parsed_addr:
                street_name_parts.append(parsed_addr['StreetName'].upper())
            
            if 'StreetNamePostType' in parsed_addr:
                suffix_raw = parsed_addr['StreetNamePostType'].upper()
                street_name_parts.append(self.street_suffix_map.get(suffix_raw, suffix_raw))
            
            if 'StreetNamePostDirectional' in parsed_addr:
                dir_raw = parsed_addr['StreetNamePostDirectional'].upper()
                street_name_parts.append(self.directional_map.get(dir_raw, dir_raw))
            
            std_record['street_name'] = ' '.join(street_name_parts)
            
            # Standardize secondary unit
            unit_type = ''
            unit_num = ''
            
            if 'OccupancyType' in parsed_addr:
                unit_type_raw = parsed_addr['OccupancyType'].upper()
                unit_type = self.unit_type_map.get(unit_type_raw, unit_type_raw)
            
            if 'OccupancyIdentifier' in parsed_addr:
                unit_num = parsed_addr['OccupancyIdentifier'].upper()
            
            if unit_type or unit_num:
                std_record['unit'] = f"{unit_type} {unit_num}".strip()
            
            # Standardize city, state, and ZIP
            if 'PlaceName' in parsed_addr:
                std_record['city'] = parsed_addr['PlaceName'].upper()
            
            if 'StateName' in parsed_addr:
                std_record['state'] = parsed_addr['StateName'].upper()
            
            if 'ZipCode' in parsed_addr:
                std_record['zip'] = parsed_addr['ZipCode']
                # Extract 5-digit ZIP if it's ZIP+4
                if '-' in std_record['zip']:
                    std_record['zip5'] = std_record['zip'].split('-')[0]
                else:
                    std_record['zip5'] = std_record['zip']
        
        except Exception as e:
            # Fallback for parsing failures
            print(f"Address parsing error: {e}")
            std_record['parsed'] = False
            return std_record
        
        # Standardize facility name
        if 'name' in record:
            std_record['name'] = self.standardize_facility_name(record['name'])
        
        # Pass through Auth ID (assuming it's already standardized)
        if 'auth_id' in record:
            std_record['auth_id'] = record['auth_id']
        
        std_record['parsed'] = True
        return std_record
    
    def standardize_facility_name(self, name: str) -> str:
        """Standardize healthcare facility names"""
        if not name:
            return ""
        
        # Convert to uppercase
        std_name = name.upper()
        
        # Replace common healthcare terms
        for term, replacement in self.healthcare_term_map.items():
            std_name = re.sub(r'\b' + term + r'\b', replacement, std_name)
        
        # Remove common words that don't add much value for matching
        stop_words = ['THE', 'OF', 'AND', 'FOR']
        for word in stop_words:
            std_name = re.sub(r'\b' + word + r'\b', '', std_name)
        
        # Remove excess whitespace
        std_name = re.sub(r'\s+', ' ', std_name).strip()
        
        return std_name

    # STEP 2: BLOCKING STRATEGY
    def create_blocking_keys(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create multiple blocking keys for efficient record comparison
        """
        blocks = {}
        
        for record in records:
            if not record.get('parsed', False):
                continue
            
            # Create multiple blocking keys for each record
            keys = []
            
            # Blocking by ZIP code (if available)
            if 'zip5' in record:
                keys.append(f"zip:{record['zip5']}")
            
            # Blocking by first 3 chars of facility name (if available)
            if 'name' in record and len(record['name']) >= 3:
                keys.append(f"name:{record['name'][:3]}")
            
            # Blocking by first char of street name + first 3 digits of street number
            if 'street_name' in record and 'street_number' in record:
                street_first = record['street_name'][:1] if record['street_name'] else ''
                num_prefix = record['street_number'][:3] if record['street_number'] else ''
                if street_first and num_prefix:
                    keys.append(f"addr:{street_first}{num_prefix}")
            
            # Add the record to each of its block keys
            for key in keys:
                if key not in blocks:
                    blocks[key] = []
                blocks[key].append(record)
        
        return blocks

    # STEP 3: RECORD COMPARISON
    def calculate_comparison_features(self, parent: Dict, atom: Dict) -> List[float]:
        """
        Calculate comparison features between a parent POC record and an atom record
        """
        features = []
        
        # 1. Exact match on ZIP5
        zip_match = 1.0 if parent.get('zip5') == atom.get('zip5') else 0.0
        features.append(zip_match)
        
        # 2. String similarity for street name
        street_name_sim = fuzz.token_sort_ratio(
            parent.get('street_name', ''), 
            atom.get('street_name', '')
        ) / 100.0
        features.append(street_name_sim)
        
        # 3. Exact match on street number
        street_num_match = 1.0 if parent.get('street_number') == atom.get('street_number') else 0.0
        features.append(street_num_match)
        
        # 4. String similarity for unit/suite
        unit_sim = fuzz.token_sort_ratio(
            parent.get('unit', ''), 
            atom.get('unit', '')
        ) / 100.0
        features.append(unit_sim)
        
        # 5. String similarity for city name
        city_sim = fuzz.token_sort_ratio(
            parent.get('city', ''), 
            atom.get('city', '')
        ) / 100.0
        features.append(city_sim)
        
        # 6. Exact match on state
        state_match = 1.0 if parent.get('state') == atom.get('state') else 0.0
        features.append(state_match)
        
        # 7. Facility name similarity
        name_sim = fuzz.token_set_ratio(
            parent.get('name', ''), 
            atom.get('name', '')
        ) / 100.0
        features.append(name_sim)
        
        # 8. Exact match on Auth ID (if available)
        auth_match = 1.0 if parent.get('auth_id') == atom.get('auth_id') else 0.0
        features.append(auth_match)
        
        # 9. Levenshtein ratio for full address comparison
        full_address_parent = f"{parent.get('street_number', '')} {parent.get('street_name', '')} {parent.get('unit', '')}"
        full_address_atom = f"{atom.get('street_number', '')} {atom.get('street_name', '')} {atom.get('unit', '')}"
        address_sim = fuzz.ratio(full_address_parent, full_address_atom) / 100.0
        features.append(address_sim)
        
        # 10. Geospatial proximity feature
        distance = self.calculate_geospatial_distance(parent, atom)
        if distance is not None:
            # Convert distance to a similarity score (closer = higher score)
            # Use exponential decay: 1.0 for 0 meters, ~0.5 for 200 meters, close to 0 for 1000+ meters
            geo_sim = math.exp(-distance / 500)  # Adjust the denominator to change sensitivity
        else:
            # If distance cannot be calculated, use neutral value
            geo_sim = 0.5
        features.append(geo_sim)
        
        return features

    # STEP 4: MACHINE LEARNING APPROACH FOR CLASSIFICATION
    def train_matching_model(self, training_data: List[Tuple[Dict, Dict, int]]) -> None:
        """
        Train a machine learning model for classification
        
        Args:
            training_data: List of (parent, atom, is_match) tuples where is_match is 1 for match, 0 for non-match
        """
        X = []
        y = []
        
        for parent, atom, is_match in training_data:
            features = self.calculate_comparison_features(parent, atom)
            X.append(features)
            y.append(is_match)
        
        # Train a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save the model
        with open('address_linkage_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
    
    def predict_match_probability(self, parent: Dict, atom: Dict) -> float:
        """Predict the probability of a match between parent and atom records"""
        if not self.model:
            raise ValueError("Model not trained. Call train_matching_model first.")
        
        features = self.calculate_comparison_features(parent, atom)
        proba = self.model.predict_proba([features])[0][1]  # Probability of class 1 (match)
        return proba

    # STEP 5: IMPLEMENTATION OF FULL RECORD LINKAGE PIPELINE
    def link_records(self, poc_parents: List[Dict], inventory_atoms: List[Dict], use_geocoding=True) -> Dict:
        """
        Main record linkage function that processes POC parents and inventory atoms
        
        Args:
            poc_parents: List of parent POC records
            inventory_atoms: List of inventory atom records
            use_geocoding: Whether to geocode addresses for geospatial validation
        
        Returns a dictionary with definite matches, possible matches, and statistics
        """
        # Step 1: Standardize all records
        print("Standardizing records...")
        standardized_parents = [self.standardize_address(record) for record in poc_parents]
        standardized_atoms = [self.standardize_address(record) for record in inventory_atoms]
        
        # Step 1b: Geocode addresses if enabled
        if use_geocoding:
            print("Geocoding addresses...")
            geocoded_parents = []
            for record in standardized_parents:
                geocoded_record = self.geocode_record(record)
                geocoded_parents.append(geocoded_record)
                # Respect rate limits for geocoding services
                time.sleep(0.2)
            
            geocoded_atoms = []
            for record in standardized_atoms:
                geocoded_record = self.geocode_record(record)
                geocoded_atoms.append(geocoded_record)
                # Respect rate limits for geocoding services
                time.sleep(0.2)
            
            # Replace standardized records with geocoded ones
            standardized_parents = geocoded_parents
            standardized_atoms = geocoded_atoms
        
        # Step 2: Create blocking keys
        print("Creating blocking keys...")
        parent_blocks = self.create_blocking_keys(standardized_parents)
        atom_blocks = self.create_blocking_keys(standardized_atoms)
        
        # Step 3: Compare records within blocks
        print("Comparing records within blocks...")
        candidate_pairs = []
        
        # Track pairs we've already compared to avoid duplicates
        compared_pairs = set()
        comparison_count = 0
        
        # Find overlapping block keys
        all_block_keys = set(parent_blocks.keys()) & set(atom_blocks.keys())
        
        for block_key in all_block_keys:
            for parent in parent_blocks[block_key]:
                for atom in atom_blocks[block_key]:
                    # Skip if we've already compared this pair
                    pair_id = (parent.get('id', ''), atom.get('id', ''))
                    if pair_id in compared_pairs:
                        continue
                    
                    compared_pairs.add(pair_id)
                    
                    # Calculate comparison features
                    features = self.calculate_comparison_features(parent, atom)
                    candidate_pairs.append((parent, atom, features))
                    comparison_count += 1
        
        print(f"Generated {len(candidate_pairs)} candidate pairs to evaluate")
        
        # Step 4: Classify matches
        print("Classifying matches...")
        if not self.model:
            raise ValueError("Model not trained. Cannot classify matches.")
        
        matches = []
        possible_matches = []
        
        for parent, atom, features in candidate_pairs:
            score = self.model.predict_proba([features])[0][1]  # Probability of match
            if score > self.HIGH_THRESHOLD:
                matches.append((parent, atom, score))
            elif score > self.LOW_THRESHOLD:
                possible_matches.append((parent, atom, score))
        
        # Step 5: Process results
        results = {
            'definite_matches': matches,
            'possible_matches': possible_matches,
            'statistics': {
                'total_parents': len(standardized_parents),
                'total_atoms': len(standardized_atoms),
                'comparison_count': comparison_count,
                'definite_match_count': len(matches),
                'possible_match_count': len(possible_matches)
            }
        }
        
        # Step 6: Generate human-readable results
        print("Generating human-readable results...")
        readable_matches = self._generate_readable_results(matches, possible_matches)
        results['readable_matches'] = readable_matches
        
        return results
    
    def _generate_readable_results(self, definite_matches, possible_matches):
        """Convert match results to a more human-readable format"""
        readable = {
            'definite_matches': [],
            'possible_matches': []
        }
        
        # Process definite matches
        for parent, atom, score in definite_matches:
            match_info = {
                'parent_id': parent.get('id', 'unknown'),
                'parent_name': parent.get('name', 'unknown'),
                'parent_address': parent.get('address', 'unknown'),
                'atom_id': atom.get('id', 'unknown'),
                'atom_name': atom.get('name', 'unknown'),
                'atom_address': atom.get('address', 'unknown'),
                'match_score': round(score, 4),
                'match_reasons': self._explain_match(parent, atom, score)
            }
            readable['definite_matches'].append(match_info)
        
        # Process possible matches
        for parent, atom, score in possible_matches:
            match_info = {
                'parent_id': parent.get('id', 'unknown'),
                'parent_name': parent.get('name', 'unknown'),
                'parent_address': parent.get('address', 'unknown'),
                'atom_id': atom.get('id', 'unknown'),
                'atom_name': atom.get('name', 'unknown'),
                'atom_address': atom.get('address', 'unknown'),
                'match_score': round(score, 4),
                'match_reasons': self._explain_match(parent, atom, score)
            }
            readable['possible_matches'].append(match_info)
        
        return readable
    
    # GEOSPATIAL VALIDATION METHODS
    def geocode_address_smarty(self, record: Dict) -> Dict:
        """
        Geocode an address using SmartyStreets API
        Returns latitude and longitude if successful
        """
        if not self.use_smartystreets:
            return record
        
        # Check if we've already geocoded this address
        address_str = record.get('address', '')
        if address_str in self.geocode_cache:
            result = self.geocode_cache[address_str]
            record.update(result)
            return record
        
        # Construct the API URL
        base_url = "https://us-street.api.smartystreets.com/street-address"
        
        # Prepare query parameters
        params = {
            'auth-id': self.smarty_auth_id,
            'auth-token': self.smarty_auth_token,
            'street': address_str
        }
        
        # Add city, state, zip if available
        if 'city' in record:
            params['city'] = record['city']
        if 'state' in record:
            params['state'] = record['state']
        if 'zip5' in record:
            params['zipcode'] = record['zip5']
        
        try:
            # Make the API request
            response = requests.get(base_url, params=params)
            data = response.json()
            
            # Process the response
            if data and len(data) > 0:
                metadata = data[0].get('metadata', {})
                if 'latitude' in metadata and 'longitude' in metadata:
                    result = {
                        'latitude': metadata['latitude'],
                        'longitude': metadata['longitude'],
                        'geocoded': True,
                        'geocoder': 'smartystreets'
                    }
                    
                    # Cache the result
                    self.geocode_cache[address_str] = result
                    
                    # Update the record
                    record.update(result)
                    return record
        
        except Exception as e:
            print(f"SmartyStreets geocoding error: {e}")
        
        # If we get here, geocoding failed
        return record
    
    def geocode_address_nominatim(self, record: Dict) -> Dict:
        """
        Geocode an address using Nominatim (OpenStreetMap)
        Falls back to this if SmartyStreets is not available or fails
        """
        # Check if we've already geocoded this address
        address_str = record.get('address', '')
        if address_str in self.geocode_cache:
            result = self.geocode_cache[address_str]
            record.update(result)
            return record
        
        # Construct a well-formatted address for Nominatim
        address_components = []
        
        if 'street_number' in record and 'street_name' in record:
            address_components.append(f"{record['street_number']} {record['street_name']}")
        elif 'address' in record:
            address_components.append(record['address'])
        
        if 'city' in record:
            address_components.append(record['city'])
        
        if 'state' in record:
            address_components.append(record['state'])
        
        if 'zip5' in record:
            address_components.append(record['zip5'])
        
        # Add USA as the country
        address_components.append("USA")
        
        # Join components into a single string
        query = ", ".join(address_components)
        
        try:
            # Query Nominatim with rate limiting
            location = self.geocoder.geocode(query)
            time.sleep(1)  # Respect usage policy: max 1 request per second
            
            if location:
                result = {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'geocoded': True,
                    'geocoder': 'nominatim'
                }
                
                # Cache the result
                self.geocode_cache[address_str] = result
                
                # Update the record
                record.update(result)
                return record
        
        except Exception as e:
            print(f"Nominatim geocoding error: {e}")
        
        # If we get here, geocoding failed
        record['geocoded'] = False
        return record
    
    def geocode_record(self, record: Dict) -> Dict:
        """
        Geocode a record using available services (SmartyStreets first, then Nominatim)
        """
        # Try SmartyStreets first if available
        if self.use_smartystreets:
            result = self.geocode_address_smarty(record)
            if result.get('geocoded', False):
                return result
        
        # Fall back to Nominatim
        return self.geocode_address_nominatim(record)
    
    def calculate_geospatial_distance(self, record1: Dict, record2: Dict) -> Optional[float]:
        """
        Calculate geospatial distance between two records in meters
        Returns None if either record is not geocoded
        """
        # Ensure both records have geocoding information
        if not record1.get('geocoded', False) or not record2.get('geocoded', False):
            return None
        
        # Calculate distance using geodesic (more accurate than great circle)
        try:
            point1 = (record1['latitude'], record1['longitude'])
            point2 = (record2['latitude'], record2['longitude'])
            
            # Calculate distance in meters
            distance = geodesic(point1, point2).meters
            return distance
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return None
    
    def _explain_match(self, parent, atom, score):
        """Generate human-readable explanation for why records matched"""
        reasons = []
        
        # Check for exact ZIP match
        if parent.get('zip5') == atom.get('zip5'):
            reasons.append(f"Same ZIP code: {parent.get('zip5')}")
        
        # Check for high name similarity
        name_sim = fuzz.token_set_ratio(parent.get('name', ''), atom.get('name', '')) / 100.0
        if name_sim > 0.8:
            reasons.append(f"Similar facility names ({round(name_sim * 100)}% similar)")
        
        # Check for high address similarity
        street_name_sim = fuzz.token_sort_ratio(
            parent.get('street_name', ''), 
            atom.get('street_name', '')
        ) / 100.0
        
        if street_name_sim > 0.8:
            reasons.append(f"Similar street names ({round(street_name_sim * 100)}% similar)")
        
        # Check for exact street number match
        if parent.get('street_number') == atom.get('street_number'):
            reasons.append(f"Same street number: {parent.get('street_number')}")
        
        # Check for Auth ID match
        if parent.get('auth_id') == atom.get('auth_id'):
            reasons.append(f"Matching Auth ID: {parent.get('auth_id')}")
        
        # Check for geospatial proximity if available
        distance = self.calculate_geospatial_distance(parent, atom)
        if distance is not None:
            if distance < 100:  # Within 100 meters
                reasons.append(f"Locations are very close: {distance:.1f} meters apart")
            elif distance < 500:  # Within 500 meters
                reasons.append(f"Locations are nearby: {distance:.1f} meters apart")
        
        # If no specific reasons found but high score
        if not reasons and score > 0.7:
            reasons.append("Multiple partial matches contribute to high overall confidence")
        
        return reasons

    # STEP 6: EVALUATION METRICS
    def evaluate_results(self, matches: List[Tuple], ground_truth: List[Tuple]) -> Dict:
        """
        Evaluate the performance of record linkage results against known ground truth
        
        Args:
            matches: List of (parent, atom, score) tuples returned by the linkage
            ground_truth: List of (parent_id, atom_id) tuples of known true matches
        
        Returns:
            Dictionary with precision, recall, F1 score, and error counts
        """
        # Convert ground truth to a set of (parent_id, atom_id) pairs
        true_matches = set(ground_truth)
        
        # Convert our predicted matches to the same format
        predicted_matches = set(
            (parent.get('id', ''), atom.get('id', '')) 
            for parent, atom, _ in matches
        )
        
        # Calculate true positives, false positives, false negatives
        true_positives = len(predicted_matches & true_matches)
        false_positives = len(predicted_matches - true_matches)
        false_negatives = len(true_matches - predicted_matches)
        
        # Calculate precision, recall, F1
        precision = true_positives / max(len(predicted_matches), 1)
        recall = true_positives / max(len(true_matches), 1)
        f1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }


# Example usage
def main():
    # You would get these from environment variables or configuration
    smarty_auth_id = None  # "YOUR_SMARTY_AUTH_ID"
    smarty_auth_token = None  # "YOUR_SMARTY_AUTH_TOKEN"
    
    # Sample data (would be loaded from CSV/database in practice)
    poc_parents = [
        {
            'id': 'P001',
            'name': 'Memorial Hospital',
            'address': '123 Main St E, Suite 400, Springfield, IL 62701',
            'auth_id': 'MEM12345'
        },
        {
            'id': 'P002',
            'name': 'North Medical Center',
            'address': '555 Oak Avenue, Chicago, IL 60601',
            'auth_id': 'NMC98765'
        }
    ]
    
    inventory_atoms = [
        {
            'id': 'A001',
            'name': 'Memorial Hospital - Main Campus',
            'address': '123 E Main Street, STE 400, Springfield, IL 62701',
            'auth_id': 'MEM12345'
        },
        {
            'id': 'A002',
            'name': 'Memorial Hospital Lab',
            'address': '123 E Main St, Suite 410, Springfield, IL 62701',
            'auth_id': 'MEM12345'
        },
        {
            'id': 'A003',
            'name': 'Northwestern Medical Center',
            'address': '555 Oak Ave, Chicago, IL 60601',
            'auth_id': 'NMC98765'
        }
    ]
    
    # Create training data (typically would be labeled manually)
    training_data = [
        (poc_parents[0], inventory_atoms[0], 1),  # Match
        (poc_parents[0], inventory_atoms[1], 1),  # Match (different department)
        (poc_parents[0], inventory_atoms[2], 0),  # Non-match
        (poc_parents[1], inventory_atoms[2], 1),  # Match
        (poc_parents[1], inventory_atoms[0], 0),  # Non-match
    ]
    
    # Ground truth for evaluation
    ground_truth = [
        ('P001', 'A001'),
        ('P001', 'A002'),
        ('P002', 'A003')
    ]
    
    # Initialize the record linkage system with optional SmartyStreets credentials
    linkage = USAddressRecordLinkage(
        model_path=None,  # Path to pre-trained model if available
        smarty_auth_id=smarty_auth_id,
        smarty_auth_token=smarty_auth_token
    )
    
    # Train the model
    print("Training model...")
    linkage.train_matching_model(training_data)
    
    # Run the record linkage
    print("Linking records...")
    results = linkage.link_records(poc_parents, inventory_atoms)
    
    # Evaluate the results
    evaluation = linkage.evaluate_results(results['definite_matches'], ground_truth)
    
    # Print results
    print("\nRecord Linkage Results:")
    print(f"Definite matches found: {len(results['definite_matches'])}")
    print(f"Possible matches requiring review: {len(results['possible_matches'])}")
    
    print("\nPerformance Metrics:")
    print(f"Precision: {evaluation['precision']:.2f}")
    print(f"Recall: {evaluation['recall']:.2f}")
    print(f"F1 Score: {evaluation['f1_score']:.2f}")
    
    print("\nSample Match Results:")
    if results['readable_matches']['definite_matches']:
        sample = results['readable_matches']['definite_matches'][0]
        print(f"Parent: {sample['parent_name']} at {sample['parent_address']}")
        print(f"Matched to: {sample['atom_name']} at {sample['atom_address']}")
        print(f"Match score: {sample['match_score']}")
        print(f"Match reasons: {', '.join(sample['match_reasons'])}")


if __name__ == "__main__":
    main()