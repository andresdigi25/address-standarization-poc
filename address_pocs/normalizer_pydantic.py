import logging
from typing import Dict, List, Optional, Any, Type, ClassVar
from pydantic import BaseModel, Field, create_model


class FieldMapping:
    """Helper class to store field mappings"""
    
    def __init__(self, mappings: Dict[str, Dict[str, List[str]]]):
        self.mappings = mappings
    
    def get_mapping(self, mapping_key: str = 'default') -> Dict[str, List[str]]:
        """Get field mapping by key, fallback to default if not found"""
        if mapping_key not in self.mappings:
            logging.warning(f"Mapping key '{mapping_key}' not found, using default")
            mapping_key = 'default'
        return self.mappings[mapping_key]


# Initialize field mappings
field_mappings = FieldMapping({
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
})


class BaseRecord(BaseModel):
    """Base record model with normalization functionality"""
    
    # Class variable to hold field mappings
    field_aliases: ClassVar[Dict[str, List[str]]] = {}
    
    @classmethod
    def normalize(cls, record: Dict[str, Any], default_values: Optional[Dict[str, Any]] = None) -> 'BaseRecord':
        """
        Normalize a record by mapping source field names to target field names defined in the model.
        
        Args:
            record: The record to normalize
            default_values: Optional dictionary of default values for target fields
            
        Returns:
            An instance of the model with normalized fields
        """
        # Start with default values if provided
        normalized_dict = default_values.copy() if default_values else {}
        
        # Process each source field
        for source_field, value in record.items():
            source_field_lower = source_field.lower().strip()
            
            # Check each target field and its possible aliases
            for target_field, aliases in cls.field_aliases.items():
                aliases_lower = [a.lower() for a in aliases]
                
                # Match if source matches any alias or contains it as a substring
                if (source_field_lower in aliases_lower or 
                    any(source_field_lower.find(a) >= 0 for a in aliases_lower)):
                    normalized_dict[target_field] = value
                    break
        
        # Create model instance with normalized data
        return cls(**normalized_dict)


# Define specific record models
class PersonRecord(BaseRecord):
    """Person record with standard contact information fields"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    
    # Define field mappings
    field_aliases = field_mappings.get_mapping('default')


class CustomerRecord(BaseRecord):
    """Customer record with customer-specific fields"""
    customer_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    membership_level: Optional[str] = Field(default="Basic")
    joined_date: Optional[str] = None
    
    # Define field mappings
    field_aliases = field_mappings.get_mapping('customer')


def create_dynamic_model(mapping_key: str, default_values: Optional[Dict[str, Any]] = None) -> Type[BaseRecord]:
    """
    Dynamically create a Pydantic model based on a mapping key.
    
    Args:
        mapping_key: The key to look up field mappings
        default_values: Optional default values for fields
        
    Returns:
        A dynamically created Pydantic model class
    """
    mapping = field_mappings.get_mapping(mapping_key)
    
    # Create field definitions
    fields = {
        field_name: (Optional[str], None) 
        for field_name in mapping.keys()
    }
    
    # Apply default values if provided
    if default_values:
        for field_name, default_value in default_values.items():
            if field_name in fields:
                fields[field_name] = (Optional[str], Field(default=default_value))
    
    # Create the model
    model = create_model(
        f'{mapping_key.capitalize()}Model', 
        __base__=BaseRecord,
        **fields
    )
    
    # Add field aliases
    model.field_aliases = mapping
    
    return model


# Example data - same as previous example
records = [
    {
        "First Name": "John",
        "Last": "Doe",
        "Email Address": "john.doe@example.com",
        "Phone Number": "555-123-4567",
        "Street Address": "123 Main St",
        "City": "Anytown",
        "State": "CA",
        "ZIP": "90210"
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
    # Example 1: Using predefined PersonRecord model
    print("Normalizing general address records with PersonRecord model:")
    print("-" * 50)
    for i, record in enumerate(records):
        normalized = PersonRecord.normalize(record)
        print(f"Record {i+1}:")
        print(f"Original: {record}")
        print(f"Normalized: {normalized.model_dump()}")
        print()
    
    # Example 2: Using predefined CustomerRecord model
    print("\nNormalizing customer records with CustomerRecord model:")
    print("-" * 50)
    for i, record in enumerate(customer_records):
        normalized = CustomerRecord.normalize(record)
        print(f"Customer {i+1}:")
        print(f"Original: {record}")
        print(f"Normalized: {normalized.model_dump()}")
        print()
    
    # Example 3: Using dynamically created model
    print("\nNormalizing records with dynamically created model:")
    print("-" * 50)
    DynamicModel = create_dynamic_model('default', {'zip_code': 'UNKNOWN'})
    normalized = DynamicModel.normalize(records[0])
    print(f"Original: {records[0]}")
    print(f"Normalized: {normalized.model_dump()}")
    
    # Example 4: Demonstrating validation by creating invalid data
    print("\nDemonstrating Pydantic validation:")
    print("-" * 50)
    try:
        # Create a class with a field that must be an email
        class ValidatedRecord(BaseRecord):
            first_name: str
            email: str  # This will be validated as an email
            
        # This will fail validation because it has a required field missing
        ValidatedRecord(email="not-an-email")
    except Exception as e:
        print(f"Validation error (as expected): {e}")


if __name__ == "__main__":
    main()