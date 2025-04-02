from datetime import date
import re
from typing import Dict, Any, List, Tuple, Optional

from app.models.atom import Atom


class DataValidator:
    """Service to validate data against the Atom model"""
    
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Validate a single record against the Atom model
        
        Args:
            record: Dictionary containing field values
            
        Returns:
            Tuple containing (is_valid, error_details)
        """
        errors = {}
        
        # Print the record for debugging
        print(f"Validating record: {record}")
        
        # Validate required fields
        required_fields = [
            "source", "facility_name", "addr1", "city", "state", 
            "zip", "auth_type", "auth_id", "data_type", "class_of_trade"
        ]
        
        for field in required_fields:
            if field not in record or record[field] is None or (isinstance(record[field], str) and record[field].strip() == ""):
                errors[field] = f"Required field '{field}' is missing or empty"
                
        # If any required fields are missing, return early
        if errors:
            print(f"Validation errors on required fields: {errors}")
            return False, errors
        
        # Validate field formats
        
        # State should be a 2-character code
        if len(record.get("state", "")) != 2:
            errors["state"] = f"State must be a 2-character code, got: '{record.get('state')}'"
        
        # ZIP code format (US format as an example)
        zip_value = record.get("zip", "")
        # Convert to string if it's a number
        if isinstance(zip_value, (int, float)):
            zip_value = str(int(zip_value))
            record["zip"] = zip_value
            
        if not re.match(r'^\d{5}(-\d{4})?$', str(zip_value)):
            errors["zip"] = f"ZIP code must be in format 12345 or 12345-6789, got: '{zip_value}'"
        
        # Date validations
        for date_field in ["expire_date", "first_observed"]:
            if date_field in record and record[date_field] is not None:
                if not isinstance(record[date_field], date):
                    errors[date_field] = f"{date_field} must be a valid date, got: '{record[date_field]}'"
        
        # Check if there are any errors
        is_valid = len(errors) == 0
        
        if not is_valid:
            print(f"Validation errors on format: {errors}")
        
        return is_valid, errors if not is_valid else None
    
    @staticmethod
    def validate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate multiple records against the Atom model
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Tuple containing (valid_records, invalid_records)
        """
        valid_records = []
        invalid_records = []
        
        print(f"Validating {len(records)} records")
        for i, record in enumerate(records):
            print(f"\nRecord {i+1}:")
            for key, value in record.items():
                print(f"  {key}: {value} (type: {type(value).__name__})")
            
            is_valid, errors = DataValidator.validate_record(record)
            
            if is_valid:
                valid_records.append(record)
                print(f"  Result: VALID")
            else:
                invalid_records.append({**record, "validation_errors": errors})
                print(f"  Result: INVALID - {errors}")
        
        print(f"Validation complete: {len(valid_records)} valid, {len(invalid_records)} invalid")
        return valid_records, invalid_records
    
    @staticmethod
    def create_atom_instances(valid_records: List[Dict[str, Any]]) -> List[Atom]:
        """
        Create Atom model instances from validated records
        
        Args:
            valid_records: List of validated record dictionaries
            
        Returns:
            List of Atom model instances
        """
        return [Atom(**record) for record in valid_records]