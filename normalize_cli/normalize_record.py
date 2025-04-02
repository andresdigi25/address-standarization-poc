
import json
import logging
from typing import Dict, Any, Tuple
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
    mapping = settings.FIELD_MAPPINGS.get(mapping_key, settings.FIELD_MAPPINGS['default'])
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
                    audit_log["matched"].append({
                        "source_field": source_field,
                        "target_field": target_field,
                        "value": value,
                        "action": "mapped"
                    })
                    logger.debug(f"Mapped: {source_field} -> {target_field} (value: {value})")
                else:
                    audit_log["matched"].append({
                        "source_field": source_field,
                        "target_field": target_field,
                        "value": value,
                        "action": "skipped (already set)"
                    })
                    logger.debug(f"Skipped overwrite: {source_field} -> {target_field} (already set)")
                matched = True
                break

        if not matched:
            audit_log["unmatched"].append({
                "source_field": source_field,
                "value": value
            })
            logger.debug(f"No match: {source_field} (value: {value})")

    with open("normalize_audit.json", "a") as f:
        f.write(json.dumps(audit_log, indent=2) + "\n")

    return normalized_record, audit_log
