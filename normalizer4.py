import json
import logging
from typing import Dict, Any, Tuple, List, Optional, Set
from datetime import datetime, timezone
from functools import lru_cache
import re

# Simulated settings module
class Settings:
    FIELD_MAPPINGS = {
        'default': {
            'first_name': ['first', 'first_name', 'fname', 'firstname', 'given_name'],
            'last_name': ['last', 'last_name', 'surname', 'lastname', 'family_name'],
            'email': ['email', 'email_address', 'e-mail', 'emailaddress', 'mail'],
            'phone': ['phone', 'phone_number', 'telephone', 'cell', 'mobile'],
            'address': ['address', 'street_address', 'mailing_address', 'residence']
        },
        'customer': {
            'customer_id': ['cust_id', 'customer_id', 'customerid', 'id', 'customer_number'],
            'loyalty_level': ['loyalty', 'status', 'membership_level', 'tier']
        }
    }
    
    # Value transformations
    VALUE_NORMALIZERS = {
        'email': lambda x: str(x).lower().strip() if x else None,
        'phone': lambda x: re.sub(r'[^0-9+]', '', str(x)) if x else None,
    }
    
    # Default values when field is missing
    DEFAULT_VALUES = {
        'loyalty_level': 'standard'
    }
    
    LOG_FILE = 'normalize_audit.log'
    JSON_AUDIT_FILE = 'normalize_audit.json'
    
    # Logging level
    LOG_LEVEL = logging.INFO
    
    # Schema validation
    REQUIRED_FIELDS = {
        'default': ['email'],
        'customer': ['customer_id', 'email']
    }

# Configure file logger
def setup_logging(log_file: str = None, log_level: int = None):
    log_file = log_file or Settings.LOG_FILE
    log_level = log_level or Settings.LOG_LEVEL
    
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Also create a console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger = logging.getLogger(__name__)
    logger.addHandler(console)
    
    return logger

logger = setup_logging()

@lru_cache(maxsize=128)
def get_mapping(mapping_key: str) -> Dict[str, List[str]]:
    """
    Get field mapping with caching for performance.
    This helps avoid repeatedly accessing the same mapping.
    """
    return Settings.FIELD_MAPPINGS.get(mapping_key, Settings.FIELD_MAPPINGS['default'])

def build_reverse_mapping(mapping: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build a reverse mapping from source field to target field for quick lookups.
    """
    reverse_map = {}
    for target, sources in mapping.items():
        for source in sources:
            source_clean = source.lower().strip()
            reverse_map[source_clean] = target
    return reverse_map

def normalize_value(field_name: str, value: Any) -> Any:
    """
    Apply any value-specific transformations based on field type.
    """
    if value is None:
        return None
        
    # Apply field-specific normalizers
    if field_name in Settings.VALUE_NORMALIZERS:
        return Settings.VALUE_NORMALIZERS[field_name](value)
    
    # Default normalization for string values
    if isinstance(value, str):
        return value.strip()
    
    return value

def normalize_record(
    record: Dict[str, Any], 
    mapping_key: str = 'default',
    apply_defaults: bool = True,
    validate_required: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize a record using a field mapping.
    
    Args:
        record: Dictionary with source field names and values
        mapping_key: Key to select the mapping definition
        apply_defaults: Whether to apply default values for missing fields
        validate_required: Whether to validate required fields
    
    Returns:
        - normalized_record: Dict with normalized keys and values
        - audit_log: Dict containing matched and unmatched fields
    """
    mapping = get_mapping(mapping_key)
    
    # Build a reverse mapping for quicker lookups
    reverse_map = build_reverse_mapping(mapping)
    
    # Combine mappings if mapping_key is not 'default'
    if mapping_key != 'default':
        default_mapping = get_mapping('default')
        reverse_default = build_reverse_mapping(default_mapping)
        # Only add default mappings that don't conflict
        for source, target in reverse_default.items():
            if source not in reverse_map:
                reverse_map[source] = target
    
    # Initialize target fields
    normalized_record = {}
    audit_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mapping_key": mapping_key,
        "record_id": record.get('id', None),  # Capture record ID if available
        "matched": [],
        "unmatched": [],
        "defaults_applied": [],
        "validation_errors": []
    }

    # First pass: match fields and normalize values
    for source_field, value in record.items():
        source_clean = source_field.strip().lower()
        
        if source_clean in reverse_map:
            target_field = reverse_map[source_clean]
            
            # Only map if not already set or if new value is not None
            if target_field not in normalized_record or (value is not None and normalized_record[target_field] is None):
                # Normalize the value based on field type
                normalized_value = normalize_value(target_field, value)
                
                # Check for meaningful change in value
                if target_field in normalized_record:
                    action = "overwritten"
                    old_value = normalized_record[target_field]
                else:
                    action = "mapped"
                    old_value = None
                
                normalized_record[target_field] = normalized_value
                
                log_entry = {
                    "source_field": source_field,
                    "target_field": target_field,
                    "value": normalized_value,
                    "action": action
                }
                
                if old_value is not None:
                    log_entry["old_value"] = old_value
                
                audit_log["matched"].append(log_entry)
                logger.debug(f"{action.capitalize()}: {source_field} -> {target_field} (value: {normalized_value})")
            else:
                log_entry = {
                    "source_field": source_field,
                    "target_field": target_field,
                    "value": value,
                    "current_value": normalized_record[target_field],
                    "action": "skipped (already set)"
                }
                audit_log["matched"].append(log_entry)
                logger.debug(f"Skipped overwrite: {source_field} -> {target_field} (already set)")
        else:
            audit_log["unmatched"].append({
                "source_field": source_field,
                "value": value
            })
            logger.debug(f"No match: {source_field} (value: {value})")

    # Second pass: apply defaults for missing fields
    if apply_defaults:
        combined_defaults = {}
        
        # Apply defaults from both default and specific mapping
        if mapping_key != 'default':
            combined_defaults.update(Settings.DEFAULT_VALUES)
        
        for field, default_value in combined_defaults.items():
            if field not in normalized_record or normalized_record[field] is None:
                normalized_record[field] = default_value
                audit_log["defaults_applied"].append({
                    "field": field,
                    "value": default_value
                })
                logger.debug(f"Applied default: {field} = {default_value}")

    # Third pass: validate required fields
    if validate_required:
        required_fields = Settings.REQUIRED_FIELDS.get(mapping_key, [])
        
        # Combine with default required fields if using a non-default mapping
        if mapping_key != 'default':
            required_fields = list(set(required_fields).union(Settings.REQUIRED_FIELDS.get('default', [])))
        
        for field in required_fields:
            if field not in normalized_record or normalized_record[field] is None:
                audit_log["validation_errors"].append({
                    "field": field,
                    "error": "required field missing"
                })
                logger.warning(f"Validation error: Required field '{field}' is missing")

    # Write JSON audit to file
    with open(Settings.JSON_AUDIT_FILE, "a") as f:
        f.write(json.dumps(audit_log, indent=2) + "\n")

    return normalized_record, audit_log


from colorama import Fore, Style, init
init(autoreset=True)

def run_tests():
    """Run test normalizations with sample data"""
    test_records = [
        {
            "FName": "Alice",
            "Surname": "Smith",
            "Email_Address": "alice@EXAMPLE.com"  # Mixed case email
        },
        {
            "first": "Bob",
            "Last_Name": "Johnson",
            "e-mail": "bob@example.com",
            "Extra_Field": "should be ignored",
            "phone": "123-456-7890"  # Phone number with separators
        },
        {
            "fname": "Charlie",
            "first": "ShouldNotOverride",
            "surname": "Brown",
            "email": "charlie@example.com"
        },
        {
            "username": "delta",
            "phone_number": "(555) 123-4567",  # Different phone format
            "city": "New York"
        }
    ]

    customer_records = [
        {
            "first_name": "Emma",
            "last_name": "Davis",
            "email": "emma@example.com",
            "cust_id": "C12345",
            "tier": "gold"
        },
        {
            "first_name": "Frank",
            "last_name": "Wilson",
            "email": "frank@example.com",
            "customer_id": "C67890"
            # Missing loyalty level - should get default
        }
    ]

    stats = {
        "total_records": 0,
        "total_matched": 0,
        "total_skipped": 0,
        "total_unmatched": 0,
        "total_defaults": 0,
        "total_validation_errors": 0
    }

    print(Fore.CYAN + "\n=== Testing Default Mapping ===")
    for i, record in enumerate(test_records, start=1):
        stats["total_records"] += 1
        print(Fore.BLUE + f"\n=== Test Record #{i} ===")
        normalized, audit = normalize_record(record)

        print(Fore.BLUE + "Normalized Record:")
        print(json.dumps(normalized, indent=2))

        print(Fore.BLUE + "\nAudit Log:")
        
        for entry in audit["matched"]:
            action = entry["action"]
            msg = f"{entry['source_field']} → {entry['target_field']} = {entry['value']}"
            
            if action == "mapped":
                print(Fore.GREEN + f"  ✅ {msg}")
                stats["total_matched"] += 1
            elif action == "overwritten":
                print(Fore.MAGENTA + f"  🔄 {msg} (overwrote: {entry.get('old_value')})")
                stats["total_matched"] += 1
            else:
                print(Fore.YELLOW + f"  ⚠️  {msg} (skipped overwrite)")
                stats["total_skipped"] += 1

        for entry in audit["unmatched"]:
            print(Fore.RED + f"  ❌ {entry['source_field']} = {entry['value']} (no match)")
            stats["total_unmatched"] += 1
            
        for entry in audit["defaults_applied"]:
            print(Fore.CYAN + f"  📝 {entry['field']} = {entry['value']} (default applied)")
            stats["total_defaults"] += 1
            
        for entry in audit["validation_errors"]:
            print(Fore.RED + f"  ⛔ {entry['field']}: {entry['error']}")
            stats["total_validation_errors"] += 1

    print(Fore.CYAN + "\n=== Testing Customer Mapping ===")
    for i, record in enumerate(customer_records, start=1):
        stats["total_records"] += 1
        print(Fore.BLUE + f"\n=== Customer Record #{i} ===")
        normalized, audit = normalize_record(record, mapping_key="customer")

        print(Fore.BLUE + "Normalized Record:")
        print(json.dumps(normalized, indent=2))
        
        print(Fore.BLUE + "\nAudit Log:")
        
        for entry in audit["matched"]:
            action = entry["action"]
            msg = f"{entry['source_field']} → {entry['target_field']} = {entry['value']}"
            
            if action == "mapped":
                print(Fore.GREEN + f"  ✅ {msg}")
                stats["total_matched"] += 1
            elif action == "overwritten":
                print(Fore.MAGENTA + f"  🔄 {msg} (overwrote: {entry.get('old_value')})")
                stats["total_matched"] += 1
            else:
                print(Fore.YELLOW + f"  ⚠️  {msg} (skipped overwrite)")
                stats["total_skipped"] += 1

        for entry in audit["unmatched"]:
            print(Fore.RED + f"  ❌ {entry['source_field']} = {entry['value']} (no match)")
            stats["total_unmatched"] += 1
            
        for entry in audit["defaults_applied"]:
            print(Fore.CYAN + f"  📝 {entry['field']} = {entry['value']} (default applied)")
            stats["total_defaults"] += 1
            
        for entry in audit["validation_errors"]:
            print(Fore.RED + f"  ⛔ {entry['field']}: {entry['error']}")
            stats["total_validation_errors"] += 1

    # Summary Stats
    print(Fore.CYAN + "\n=== Summary Statistics ===")
    print(Fore.CYAN + f"Total Records Processed:    {stats['total_records']}")
    print(Fore.GREEN + f"Total Fields Mapped:        {stats['total_matched']}")
    print(Fore.YELLOW + f"Total Fields Skipped:       {stats['total_skipped']}")
    print(Fore.RED + f"Total Fields Unmatched:     {stats['total_unmatched']}")
    print(Fore.CYAN + f"Total Defaults Applied:     {stats['total_defaults']}")
    print(Fore.RED + f"Total Validation Errors:    {stats['total_validation_errors']}")


if __name__ == "__main__":
    run_tests()