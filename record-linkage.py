def jaro_distance(s1, s2):
    """
    Calculate the Jaro distance between two strings.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        float: Jaro distance between 0 and 1
    """
    # If the strings are equal
    if s1 == s2:
        return 1.0
    
    # If either string is empty
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    # Maximum distance for matching characters
    match_distance = max(len(s1), len(s2)) // 2 - 1
    match_distance = max(0, match_distance)  # Ensure it's not negative
    
    # Initialize match and transposition counters
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    # Count matching characters
    matches = 0
    for i in range(len(s1)):
        # Calculate window boundaries
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))
        
        for j in range(start, end):
            # Skip already matched characters in s2
            if s2_matches[j]:
                continue
            
            # Skip non-matching characters
            if s1[i] != s2[j]:
                continue
            
            # Mark characters as matched
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    # If no matches were found, return 0
    if matches == 0:
        return 0.0
    
    # Count transpositions
    transpositions = 0
    k = 0
    
    for i in range(len(s1)):
        # Skip non-matched characters
        if not s1_matches[i]:
            continue
        
        # Find the next matched character in s2
        while k < len(s2) and not s2_matches[k]:
            k += 1
        
        # Check if there is a transposition
        if k < len(s2) and s1[i] != s2[k]:
            transpositions += 1
        
        k += 1
    
    # Divide transpositions by 2 (as per the algorithm definition)
    transpositions = transpositions // 2
    
    # Calculate Jaro distance components
    m = matches
    
    # Calculate Jaro distance
    jaro = (m / len(s1) + m / len(s2) + (m - transpositions) / m) / 3.0
    
    return jaro

def jaro_winkler(s1, s2, p=0.1):
    """
    Calculate the Jaro-Winkler similarity between two strings.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        p (float): Scaling factor for prefix adjustments, max 0.25
        
    Returns:
        float: Jaro-Winkler similarity between 0 and 1
    """
    # Calculate base Jaro distance
    jaro_dist = jaro_distance(s1, s2)
    
    # Calculate length of common prefix (up to 4 characters)
    prefix_len = 0
    max_prefix = min(4, min(len(s1), len(s2)))
    
    for i in range(max_prefix):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    # Ensure p doesn't exceed 0.25 (as per algorithm definition)
    p = min(0.25, p)
    
    # Calculate Jaro-Winkler similarity
    jaro_winkler_sim = jaro_dist + (prefix_len * p * (1 - jaro_dist))
    
    return jaro_winkler_sim

def normalize_address(address):
    """
    Normalize an address string for better comparison.
    
    Args:
        address (str): Original address string
        
    Returns:
        str: Normalized address string
    """
    # Convert to lowercase
    address = address.lower()
    
    # Common abbreviation replacements
    abbreviations = {
        "avenue": "ave",
        "boulevard": "blvd",
        "circle": "cir",
        "court": "ct",
        "drive": "dr",
        "lane": "ln",
        "place": "pl",
        "road": "rd",
        "square": "sq",
        "street": "st",
        "terrace": "ter",
        "apartment": "apt",
        "building": "bldg",
        "floor": "fl",
        "suite": "ste",
        "unit": "unit",
        "north": "n",
        "south": "s",
        "east": "e",
        "west": "w",
        "northeast": "ne",
        "northwest": "nw",
        "southeast": "se",
        "southwest": "sw"
    }
    
    # Apply abbreviation replacements
    normalized = address
    for full, abbr in abbreviations.items():
        # Replace full word with abbreviation
        normalized = normalized.replace(f" {full} ", f" {abbr} ")
        normalized = normalized.replace(f" {full},", f" {abbr},")
        normalized = normalized.replace(f" {full}.", f" {abbr}.")
        
        # Also replace abbreviation + period with standard abbreviation
        normalized = normalized.replace(f" {abbr}.", f" {abbr} ")
    
    # Remove common punctuation
    for char in [',', '.', '#', '-', '/', '\\']:
        normalized = normalized.replace(char, ' ')
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized

def extract_address_components(address):
    """
    Extract components from address string for weighted matching.
    
    Args:
        address (str): Normalized address string
        
    Returns:
        dict: Components of the address
    """
    components = {
        'house_number': None,
        'street_name': None,
        'unit': None,
        'city': None,
        'zipcode': None
    }
    
    # Basic extraction - in a production system, you would use a more sophisticated
    # address parser like libpostal or a regex-based approach
    
    # Split address on whitespace
    parts = address.split()
    
    # Try to extract house number (usually first numeric part)
    for part in parts:
        if part.isdigit():
            components['house_number'] = part
            break
    
    # Try to extract zip code (5-digit number, typically at the end)
    for part in parts:
        if part.isdigit() and len(part) == 5:
            components['zipcode'] = part
    
    # Look for common unit identifiers
    for i, part in enumerate(parts):
        if part in ['apt', 'unit', 'ste', 'suite', '#']:
            if i+1 < len(parts):
                components['unit'] = parts[i+1]
    
    return components

def address_similarity(addr1, addr2):
    """
    Calculate similarity between two addresses using a weighted approach.
    
    Args:
        addr1 (str): First address
        addr2 (str): Second address
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Normalize both addresses
    norm_addr1 = normalize_address(addr1)
    norm_addr2 = normalize_address(addr2)
    
    # Calculate overall string similarity
    overall_similarity = jaro_winkler(norm_addr1, norm_addr2, p=0.1)
    
    # Extract components
    components1 = extract_address_components(norm_addr1)
    components2 = extract_address_components(norm_addr2)
    
    # Component-specific weights
    weights = {
        'house_number': 0.4,  # House number is very important
        'street_name': 0.3,   # Street name is important
        'unit': 0.1,          # Unit identifier has some importance
        'city': 0.1,          # City name has some importance
        'zipcode': 0.1        # ZIP code has some importance
    }
    
    # Calculate component-specific similarities
    component_similarity = 0.0
    total_weight = 0.0
    
    for component, weight in weights.items():
        comp1 = components1.get(component)
        comp2 = components2.get(component)
        
        # If both components exist, calculate similarity
        if comp1 and comp2:
            comp_sim = jaro_winkler(comp1, comp2)
            component_similarity += weight * comp_sim
            total_weight += weight
    
    # If we couldn't extract components, rely solely on overall similarity
    if total_weight == 0:
        return overall_similarity
    
    # Calculate weighted component similarity
    weighted_component_sim = component_similarity / total_weight
    
    # Final similarity is a blend of overall and component-specific similarity
    final_similarity = (overall_similarity * 0.5) + (weighted_component_sim * 0.5)
    
    return final_similarity

def link_address_records(addresses1, addresses2, threshold=0.85):
    """
    Link address records between two datasets.
    
    Args:
        addresses1 (list): First set of address records (dicts with id, address)
        addresses2 (list): Second set of address records (dicts with id, address)
        threshold (float): Similarity threshold for matches
        
    Returns:
        list: Matched record pairs with similarity scores
    """
    matches = []
    
    for record1 in addresses1:
        best_match = None
        best_score = threshold  # Only consider matches above threshold
        
        for record2 in addresses2:
            # Calculate address similarity
            similarity = address_similarity(record1['address'], record2['address'])
            
            if similarity > best_score:
                best_score = similarity
                best_match = (record2, similarity)
        
        if best_match:
            matches.append({
                'record1': record1,
                'record2': best_match[0],
                'similarity': best_score
            })
    
    return matches

# Example usage
if __name__ == "__main__":
    # Example address datasets
    dataset1 = [
        {'id': 'A1', 'address': '123 Main Street, Apt 4B, New York, NY 10001'},
        {'id': 'A2', 'address': '456 Oak Avenue, Suite 202, Los Angeles, CA 90001'},
        {'id': 'A3', 'address': '789 Pine Road, Chicago, IL 60601'},
        {'id': 'A4', 'address': '321 Cedar Boulevard, San Francisco, CA 94101'},
        {'id': 'A5', 'address': '555 Maple Lane, Boston, MA 02101'}
    ]
    
    dataset2 = [
        {'id': 'B1', 'address': '123 Main St, #4B, New York, NY 10001'},
        {'id': 'B2', 'address': '456 Oak Ave. Ste 202, Los Angeles, California 90001'},
        {'id': 'B3', 'address': '789 Pine Rd, Chicago, Illinois 60601'},
        {'id': 'B4', 'address': '321 Cedar Blvd, SF, CA 94101'},
        {'id': 'B5', 'address': '555 Maple Ln, Apt 3C, Boston, MA 02101'},
        {'id': 'B6', 'address': '999 Elm Street, Miami, FL 33101'}
    ]
    
    print("Address Record Linkage Example:")
    print("-" * 60)
    
    # Test the normalization
    print("Address Normalization Examples:")
    test_addresses = [
        "123 Main Street, Apt 4B, New York, NY 10001",
        "456 Oak Avenue, Suite 202, Los Angeles, CA 90001"
    ]
    
    for addr in test_addresses:
        print(f"Original: {addr}")
        print(f"Normalized: {normalize_address(addr)}")
        print()
    
    # Test similarity calculation
    print("Address Similarity Examples:")
    test_pairs = [
        ("123 Main Street, Apt 4B, New York, NY 10001", "123 Main St, #4B, New York, NY 10001"),
        ("456 Oak Avenue, Suite 202, Los Angeles, CA 90001", "456 Oak Ave. Ste 202, Los Angeles, California 90001"),
        ("789 Pine Road, Chicago, IL 60601", "123 Main Street, Chicago, IL 60601")
    ]
    
    for addr1, addr2 in test_pairs:
        sim = address_similarity(addr1, addr2)
        print(f"Similarity: {sim:.4f}")
        print(f"  Address 1: {addr1}")
        print(f"  Address 2: {addr2}")
        print()
    
    # Perform record linkage
    matches = link_address_records(dataset1, dataset2)
    
    print("Matched Records:")
    for match in matches:
        print(f"Match score: {match['similarity']:.4f}")
        print(f"  Dataset 1: {match['record1']['id']} - {match['record1']['address']}")
        print(f"  Dataset 2: {match['record2']['id']} - {match['record2']['address']}")
        print()