import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class Settings:
    """Mock settings class to hold field mappings configuration."""
    FIELD_MAPPINGS: Dict[str, Dict[str, List[str]]]
    
    
# Initialize settings with field mappings
settings = Settings(
    FIELD_MAPPINGS={
        'default': {
            'first_name': ['first name', 'firstname', 'fname', 'given name'],
            'last_name': ['last name', 'lastname', 'lname', 'surname', 'family name'],
            'email': ['email', 'email address', 'e-mail', 'mail'],
            'phone': ['phone', 'telephone', 'phone number', 'cell', 'mobile'],
            'address': ['address', 'street address', 'location'],
            'city': ['city', 'town'],
            'state': ['state', 'province', 'region'],
            'zip_code': ['zip', 'zip code', 'postal code', 'postcode'],
        },
        'customer': {
            'customer_id': ['id', 'customer id', 'cust id', 'client id'],
            'first_name': ['first name', 'firstname', 'fname', 'customer first name'],
            'last_name': ['last name', 'lastname', 'lname', 'customer last name'],
            'email': ['email', 'customer email', 'contact email'],
            'phone': ['phone', 'customer phone', 'contact phone'],
            'membership_level': ['level', 'tier', 'membership', 'status'],
            'joined_date': ['joined', 'date joined', 'member since', 'start date'],
        }
    }
)


def normalize_record(
    record: Dict[str, Any],
    mapping_key: str = 'default',
    default_values: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Normalize a record by mapping source field names to target field names.
    
    Args:
        record: The record to normalize
        mapping_key: The key to use to look up field mappings
        default_values: Optional dictionary of default values for target fields
    
    Returns:
        A normalized record with standardized field names
    
    Example:
        >>> normalize_record({"First Name": "John", "Last": "Doe"})
        {"first_name": "John", "last_name": "Doe", ...}
    """
    # Get the appropriate mapping
    if mapping_key not in settings.FIELD_MAPPINGS:
        logging.warning(f"Mapping key '{mapping_key}' not found, using default")
        mapping_key = 'default'
    
    mapping = settings.FIELD_MAPPINGS[mapping_key]
    
    # Initialize with default values for all target fields
    defaults = default_values or {}
    normalized_record = {target_field: defaults.get(target_field, None) 
                         for target_field in mapping.keys()}
    
    # Create lowercase versions of source fields for matching
    lowercase_mapping = {
        target: [s.lower() for s in sources] 
        for target, sources in mapping.items()
    }
    
    # Process the record fields
    for source_field, value in record.items():
        source_field_lower = source_field.lower().strip()
        
        # Check each target field
        for target_field, possible_source_fields in lowercase_mapping.items():
            # Exact match or word boundary match
            if (source_field_lower in possible_source_fields or 
                any(source_field_lower.find(f) >= 0 for f in possible_source_fields)):
                normalized_record[target_field] = value
                break  # Stop checking other target fields once a match is found
                
    return normalized_record


# Example usage with various records

# Sample records with inconsistent field names
records = [
    {
        "First Name": "John",
        "Last": "Doe",
        "Email Address": "john.doe@example.com",
        "Phone Number": "555-123-4567",
        "Street Address": "123 Main St",
        "City": "Anytown",
        "State": "CA",
        "ZIP": "90210",
        "pinga":"pingarela"
    },
    {
        "fname": "Jane",
        "surname": "Smith",
        "mail": "jane.smith@example.com",
        "mobile": "555-987-6543",
        "location": "456 Oak Ave",
        "town": "Somewhere",
        "region": "NY",
        "postcode": "10001"
    },
    {
        "given name": "Bob",
        "family name": "Johnson",
        "e-mail": "bob.johnson@example.com",
        "cell": "555-456-7890",
        "address": "789 Pine Rd",
        "city": "Nowhere",
        "province": "TX",
        "postal code": "75001"
    }
]

# Customer records with different fields
customer_records = [
    {
        "Customer ID": "C12345",
        "First Name": "Alice",
        "Last Name": "Cooper",
        "Customer Email": "alice@example.com",
        "Contact Phone": "555-111-2222",
        "Membership": "Gold",
        "Date Joined": "2023-01-15"
    },
    {
        "ID": "C67890",
        "fname": "Bob",
        "lname": "Dylan",
        "email": "bob@example.com",
        "phone": "555-333-4444",
        "Level": "Silver",
        "Member Since": "2023-03-20"
    }
]

def main():
    print("Normalizing general address records:")
    print("-" * 50)
    for i, record in enumerate(records):
        normalized = normalize_record(record)
        print(f"Record {i+1}:")
        print(f"Original: {record}")
        print(f"Normalized: {normalized}")
        print()
    
    print("\nNormalizing customer records with 'customer' mapping:")
    print("-" * 50)
    for i, record in enumerate(customer_records):
        normalized = normalize_record(record, mapping_key='customer', 
                                     default_values={'membership_level': 'Basic'})
        print(f"Customer {i+1}:")
        print(f"Original: {record}")
        print(f"Normalized: {normalized}")
        print()
    
    # Example with a non-existent mapping key (will fall back to default)
    print("\nUsing non-existent mapping key (falls back to default):")
    print("-" * 50)
    test_record = {"First Name": "Test", "Last Name": "User"}
    normalized = normalize_record(test_record, mapping_key='nonexistent')
    print(f"Original: {test_record}")
    print(f"Normalized: {normalized}")


if __name__ == "__main__":
    main()