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
        while not s2_matches[k]:
            k += 1
        
        # Check if there is a transposition
        if s1[i] != s2[k]:
            transpositions += 1
        
        k += 1
    
    # Divide transpositions by 2 (as per the algorithm definition)
    transpositions = transpositions // 2
    
    # Calculate Jaro distance components
    m = matches
    t = transpositions
    
    # Calculate Jaro distance
    jaro = (m / len(s1) + m / len(s2) + (m - t) / m) / 3.0
    
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

def deduplicate_records(records, key_field, threshold=0.85):
    """
    Deduplicate records based on string similarity of a key field.
    
    Args:
        records (list): List of dictionaries containing records
        key_field (str): The field in records to use for comparison
        threshold (float): Similarity threshold for considering records duplicates
        
    Returns:
        tuple: (unique_records, duplicate_groups)
    """
    # Initialize sets and lists for tracking
    unique_records = []
    all_duplicate_groups = []
    processed_indices = set()
    
    # Compare each record with every other record
    for i in range(len(records)):
        # Skip already processed records
        if i in processed_indices:
            continue
        
        current_group = [i]
        current_record = records[i]
        current_key = str(current_record.get(key_field, "")).lower()
        
        for j in range(i + 1, len(records)):
            # Skip already processed records
            if j in processed_indices:
                continue
                
            compare_record = records[j]
            compare_key = str(compare_record.get(key_field, "")).lower()
            
            # Calculate similarity
            similarity = jaro_winkler(current_key, compare_key)
            
            # If similarity is above threshold, consider it a duplicate
            if similarity >= threshold:
                current_group.append(j)
                processed_indices.add(j)
        
        # Add first record of group to unique records
        unique_records.append(records[i])
        
        # If duplicates were found, store the group
        if len(current_group) > 1:
            # Create group with actual records
            duplicate_group = [records[idx] for idx in current_group]
            all_duplicate_groups.append(duplicate_group)
        
        # Mark current record as processed
        processed_indices.add(i)
    
    return unique_records, all_duplicate_groups

# Example Usage
if __name__ == "__main__":
    # Simple string comparison examples
    string_pairs = [
        ("Martha", "Marhta"),
        ("Dixon", "Dicksonx"),
        ("Jellyfish", "Smellyfish"),
        ("John Smith", "Jon Smith"),
        ("New York", "New York City")
    ]
    
    print("Jaro-Winkler Similarity Examples:")
    print("-" * 50)
    for s1, s2 in string_pairs:
        similarity = jaro_winkler(s1, s2)
        print(f"'{s1}' vs '{s2}': {similarity:.4f}")
    
    print("\n")
    
    # Record deduplication example
    customer_records = [
        {"id": 1, "name": "John Smith", "email": "john.smith@example.com", "city": "New York"},
        {"id": 2, "name": "Jon Smith", "email": "jon.smith@example.com", "city": "New York"},
        {"id": 3, "name": "Jane Doe", "email": "jane.doe@example.com", "city": "Los Angeles"},
        {"id": 4, "name": "Janes Does", "email": "jane.does@example.com", "city": "Los Angelas"},
        {"id": 5, "name": "Robert Johnson", "email": "robert.j@example.com", "city": "Chicago"},
        {"id": 6, "name": "Robert Jonson", "email": "r.jonson@example.com", "city": "Chicago"},
        {"id": 7, "name": "Sarah Williams", "email": "sarah.w@example.com", "city": "Boston"},
        {"id": 8, "name": "Alex Johnson", "email": "alex.j@example.com", "city": "Denver"}
    ]
    
    print("Record Deduplication Example:")
    print("-" * 50)
    unique_records, duplicate_groups = deduplicate_records(customer_records, "name", threshold=0.90)
    
    print(f"Original record count: {len(customer_records)}")
    print(f"Unique record count: {len(unique_records)}")
    print(f"Duplicate groups found: {len(duplicate_groups)}")
    
    print("\nDuplicate Groups:")
    for i, group in enumerate(duplicate_groups):
        print(f"\nGroup {i+1}:")
        for record in group:
            print(f"  - ID: {record['id']}, Name: {record['name']}, City: {record['city']}")
    
    # Example of record linkage across two different datasets
    database_1 = [
        {"id": "DB1-001", "customer": "John Smith", "location": "New York"},
        {"id": "DB1-002", "customer": "Jane Doe", "location": "Los Angeles"},
        {"id": "DB1-003", "customer": "Robert Johnson", "location": "Chicago"}
    ]
    
    database_2 = [
        {"id": "DB2-001", "customer": "Jon Smith", "location": "NY"},
        {"id": "DB2-002", "customer": "Jane D.", "location": "LA"},
        {"id": "DB2-003", "customer": "Bob Johnson", "location": "Chicago"},
        {"id": "DB2-004", "customer": "Sarah Williams", "location": "Boston"}
    ]
    
    print("\n\nRecord Linkage Example:")
    print("-" * 50)
    print("Linking records across two databases:")
    
    matches = []
    
    for record1 in database_1:
        best_match = None
        best_score = 0
        
        for record2 in database_2:
            # Compare customer names
            similarity = jaro_winkler(record1["customer"].lower(), record2["customer"].lower())
            
            if similarity > 0.8 and similarity > best_score:
                best_score = similarity
                best_match = (record2, similarity)
        
        if best_match:
            matches.append((record1, best_match[0], best_score))
    
    print("\nMatched Records:")
    for record1, record2, score in matches:
        print(f"  - {record1['id']} '{record1['customer']}' matches {record2['id']} '{record2['customer']}' (score: {score:.4f})")