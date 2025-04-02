def levenshtein_distance(str1, str2):
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is a measure of the minimum number of single-character
    operations (insertions, deletions, or substitutions) required to change one string
    into another.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        int: The Levenshtein distance between the two strings
    """
    # Create a matrix of size (len(str1)+1) x (len(str2)+1)
    rows = len(str1) + 1
    cols = len(str2) + 1
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize the first row and column
    for i in range(rows):
        distance_matrix[i][0] = i
    
    for j in range(cols):
        distance_matrix[0][j] = j
    
    # Fill in the rest of the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if str1[i-1] == str2[j-1]:
                # If characters match, cost is 0
                cost = 0
            else:
                # If characters don't match, cost is 1
                cost = 1
            
            # Calculate the minimum cost of operations
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1] + cost
            
            # Take the minimum of the three operations
            distance_matrix[i][j] = min(deletion, insertion, substitution)
    
    # Display the matrix (for educational purposes)
    print("Levenshtein Distance Matrix:")
    print("    " + " ".join([f"{c}" for c in " " + str2]))
    for i, row in enumerate(distance_matrix):
        prefix = str1[i-1] if i > 0 else " "
        print(f"{prefix} {' '.join([f'{cell}' for cell in row])}")
    print()
    
    # Return the bottom-right cell which contains the final distance
    return distance_matrix[rows-1][cols-1]

def levenshtein_normalized(str1, str2):
    """
    Calculate the normalized Levenshtein similarity between two strings.
    
    Returns a value between 0 (completely different) and 1 (identical strings).
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        float: Normalized similarity between 0 and 1
    """
    # Calculate the raw Levenshtein distance
    distance = levenshtein_distance(str1, str2)
    
    # Normalize by the maximum possible distance (length of the longer string)
    max_len = max(len(str1), len(str2))
    
    # Avoid division by zero
    if max_len == 0:
        return 1.0  # Two empty strings are identical
    
    # Convert distance to similarity (1 - normalized distance)
    similarity = 1.0 - (distance / max_len)
    
    return similarity

# Example usage
if __name__ == "__main__":
    print("Levenshtein Distance Examples:")
    print("-" * 50)
    
    # Example 1: Simple case with substitution only
    s1 = "kitten"
    s2 = "sitting"
    distance = levenshtein_distance(s1, s2)
    similarity = levenshtein_normalized(s1, s2)
    print(f"Example 1: '{s1}' to '{s2}'")
    print(f"Distance: {distance}")
    print(f"Normalized similarity: {similarity:.4f}")
    print("Operations:")
    print("  1. Substitute 'k' with 's' (kitten -> sitten)")
    print("  2. Substitute 'e' with 'i' (sitten -> sittin)")
    print("  3. Insert 'g' (sittin -> sitting)")
    print()
    
    # Example 2: Identical strings
    s1 = "hello"
    s2 = "hello"
    distance = levenshtein_distance(s1, s2)
    similarity = levenshtein_normalized(s1, s2)
    print(f"Example 2: '{s1}' to '{s2}'")
    print(f"Distance: {distance}")
    print(f"Normalized similarity: {similarity:.4f}")
    print()
    
    # Example 3: Deletion only
    s1 = "python"
    s2 = "pyton"
    distance = levenshtein_distance(s1, s2)
    similarity = levenshtein_normalized(s1, s2)
    print(f"Example 3: '{s1}' to '{s2}'")
    print(f"Distance: {distance}")
    print(f"Normalized similarity: {similarity:.4f}")
    print("Operations:")
    print("  1. Delete 'h' (python -> pyton)")
    print()
    
    # Example 4: Insertion only
    s1 = "data"
    s2 = "database"
    distance = levenshtein_distance(s1, s2)
    similarity = levenshtein_normalized(s1, s2)
    print(f"Example 4: '{s1}' to '{s2}'")
    print(f"Distance: {distance}")
    print(f"Normalized similarity: {similarity:.4f}")
    print("Operations:")
    print("  1. Insert 'b' (data -> datab)")
    print("  2. Insert 'a' (datab -> databa)")
    print("  3. Insert 's' (databa -> databas)")
    print("  4. Insert 'e' (databas -> database)")
    print()
    
    # Example 5: Mixed operations
    s1 = "saturday"
    s2 = "sunday"
    distance = levenshtein_distance(s1, s2)
    similarity = levenshtein_normalized(s1, s2)
    print(f"Example 5: '{s1}' to '{s2}'")
    print(f"Distance: {distance}")
    print(f"Normalized similarity: {similarity:.4f}")
    print("Operations:")
    print("  1. Delete 'a' (saturday -> sturday)")
    print("  2. Delete 't' (sturday -> surday)")
    print("  3. Substitute 'r' with 'n' (surday -> sunday)")
    print()