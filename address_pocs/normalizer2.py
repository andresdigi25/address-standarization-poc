import json
import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime

# Simulated settings module
class settings:
    FIELD_MAPPINGS = {
        'default': {
            'first_name': ['first', 'first_name', 'fname'],
            'last_name': ['last', 'last_name', 'surname'],
            'email': ['email', 'email_address', 'e-mail']
        }
    }

# Configure file logger
logging.basicConfig(
    filename='normalize_audit.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def normalize_record(record: Dict[str, Any], mapping_key: str = 'default') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize a record using a field mapping.
    
    Returns:
        - normalized_record: Dict with normalized keys and values.
        - audit_log: Dict containing matched and unmatched fields.
    """
    mapping = settings.FIELD_MAPPINGS.get(mapping_key, settings.FIELD_MAPPINGS['default'])

    # Initialize target fields
    normalized_record = {target: None for target in mapping.keys()}
    audit_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "mapping_key": mapping_key,
        "matched": [],
        "unmatched": []
    }

    for source_field, value in record.items():
        source_clean = source_field.strip().lower()
        matched = False

        for target_field, source_options in mapping.items():
            source_options_clean = [s.lower().strip() for s in source_options]

            if source_clean in source_options_clean:
                if normalized_record[target_field] is None:
                    normalized_record[target_field] = value
                    log_entry = {
                        "source_field": source_field,
                        "target_field": target_field,
                        "value": value,
                        "action": "mapped"
                    }
                    audit_log["matched"].append(log_entry)
                    logger.debug(f"Mapped: {source_field} -> {target_field} (value: {value})")
                else:
                    log_entry = {
                        "source_field": source_field,
                        "target_field": target_field,
                        "value": value,
                        "action": "skipped (already set)"
                    }
                    audit_log["matched"].append(log_entry)
                    logger.debug(f"Skipped overwrite: {source_field} -> {target_field} (already set)")
                matched = True
                break  # Stop checking once matched

        if not matched:
            audit_log["unmatched"].append({
                "source_field": source_field,
                "value": value
            })
            logger.debug(f"No match: {source_field} (value: {value})")

    # Write JSON audit to file
    with open("normalize_audit.json", "a") as f:
        f.write(json.dumps(audit_log, indent=2) + "\n")

    return normalized_record, audit_log


if __name__ == "__main__":
    test_records = [
        {
            "FName": "Alice",
            "Surname": "Smith",
            "Email_Address": "alice@example.com"
        },
        {
            "first": "Bob",                       # will map to first_name
            "Last_Name": "Johnson",               # will map to last_name
            "e-mail": "bob@example.com",          # will map to email
            "Extra_Field": "should be ignored"    # no match
        },
        {
            "fname": "Charlie",                   # will map to first_name
            "first": "ShouldNotOverride",         # should be skipped (overwrite prevention)
            "surname": "Brown",                   # will map to last_name
            "email": "charlie@example.com"        # will map to email
        },
        {
            "username": "delta",                  # no match
            "phone_number": "123-456-7890",       # no match
            "city": "New York"                    # no match
        }
    ]

    for i, record in enumerate(test_records, start=1):
        print(f"\n=== Test Record #{i} ===")
        normalized, audit = normalize_record(record)
        print("Normalized Record:")
        print(json.dumps(normalized, indent=2))

        print("\nAudit Log:")
        print(json.dumps(audit, indent=2))
        break
