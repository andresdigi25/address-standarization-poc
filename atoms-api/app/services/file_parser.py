import json
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, date
from io import StringIO, BytesIO

from app.schemas.mapping import FileMapping, ColumnMapping


class FileParser:
    """Service to parse and transform uploaded files based on mappings"""
    
    @staticmethod
    def parse_file(file_content: bytes, mapping: FileMapping) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse file content according to mapping definition
        
        Args:
            file_content: Binary content of the uploaded file
            mapping: Mapping configuration
            
        Returns:
            Tuple containing (valid_records, error_records)
        """
        if mapping.file_type.lower() == "csv":
            return FileParser._parse_csv(file_content, mapping)
        elif mapping.file_type.lower() == "json":
            return FileParser._parse_json(file_content, mapping)
        else:
            raise ValueError(f"Unsupported file type: {mapping.file_type}")

    @staticmethod
    def _parse_csv(file_content: bytes, mapping: FileMapping) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse CSV file content"""
        # Load CSV into pandas DataFrame
        try:
            print(f"Parsing CSV file, first 200 bytes: {file_content[:200]}")
            
            # Explicitly set dtype to object to treat all columns as strings
            df = pd.read_csv(BytesIO(file_content), dtype=object)
            
            print(f"CSV parsed successfully, columns: {df.columns.tolist()}")
            print(f"Number of rows: {len(df)}")
            
            if not df.empty:
                print(f"First row: {df.iloc[0].to_dict()}")
            
            # Check if required columns are present
            required_columns = [col.source for col in mapping.columns if col.required]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            print(f"Required columns: {required_columns}")
            print(f"Missing columns: {missing_columns}")
            
            # Convert empty strings to None for consistency
            df = df.replace('', None)
            
        except Exception as e:
            print(f"CSV parsing error: {str(e)}")
            raise ValueError(f"Failed to parse CSV: {str(e)}")
        
        return FileParser._process_dataframe(df, mapping)
    
    @staticmethod
    def _parse_json(file_content: bytes, mapping: FileMapping) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse JSON file content"""
        try:
            # Decode bytes to string and load JSON
            data = json.loads(file_content.decode('utf-8'))
            
            # Handle different JSON structures
            if isinstance(data, list):
                # JSON is already a list of records
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # JSON might be a dict with records in a nested key
                # Try to find a list in the JSON structure
                list_found = False
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        df = pd.DataFrame(value)
                        list_found = True
                        break
                
                if not list_found:
                    # Handle single record as dict
                    df = pd.DataFrame([data])
            else:
                raise ValueError("JSON format not recognized")
                
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")
        
        return FileParser._process_dataframe(df, mapping)
    
    @staticmethod
    def _process_dataframe(df: pd.DataFrame, mapping: FileMapping) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process DataFrame by applying column mappings"""
        valid_records = []
        error_records = []
        
        # Create mapping of source column to target and rules
        column_map = {col.source: col for col in mapping.columns}
        
        # Check if required source columns are present
        required_columns = [col.source for col in mapping.columns if col.required]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns missing: {', '.join(missing_columns)}")
        
        # Process each row
        for index, row in df.iterrows():
            record = {}
            error = False
            error_details = {}
            
            # Default source field for tracking
            record["source"] = mapping.name
            
            # Map each field
            for source_col, mapping_rule in column_map.items():
                target_field = mapping_rule.target
                
                # Get value, handling the case where column might not exist
                if source_col in df.columns:
                    value = row[source_col]
                    
                    # Ensure all values are of appropriate types
                    if pd.notna(value):
                        # Convert to string for specific fields
                        if target_field in ["zip", "auth_id", "state", "source", "facility_name", "addr1", "addr2", "city", "auth_type", "data_type", "class_of_trade"]:
                            value = str(value)
                        # Handle date conversions separately in transformations
                elif mapping_rule.default is not None:
                    value = mapping_rule.default
                elif mapping_rule.required:
                    error = True
                    error_details[target_field] = f"Required column '{source_col}' missing"
                    continue
                else:
                    # Non-required field with no default, skip
                    continue
                
                # Handle NaN values
                if pd.isna(value):
                    if mapping_rule.default is not None:
                        value = mapping_rule.default
                    elif mapping_rule.required:
                        error = True
                        error_details[target_field] = f"Required value for '{source_col}' is missing"
                        continue
                    else:
                        # Keep as None for optional fields
                        record[target_field] = None
                        continue
                
                # Apply transformations if specified
                if mapping_rule.transformation:
                    try:
                        value = FileParser._apply_transformation(value, mapping_rule.transformation)
                    except Exception as e:
                        error = True
                        error_details[target_field] = f"Transformation error for '{source_col}': {str(e)}"
                        continue
                
                # Ensure numeric values are converted to strings when needed
                if target_field in ["zip", "auth_id"] and not isinstance(value, str):
                    value = str(value)
                
                # Store the value
                record[target_field] = value
            
            # Add record to appropriate list
            if error:
                error_record = {**record, "errors": error_details, "row_index": index}
                error_records.append(error_record)
            else:
                valid_records.append(record)
        
        return valid_records, error_records
    
    @staticmethod
    def _apply_transformation(value: Any, transformation: str) -> Any:
        """Apply a transformation to a value"""
        # Convert None to empty string for string operations
        if value is None and transformation in ["uppercase", "lowercase", "strip"]:
            return ""

        # Handle type conversions before transformations
        if transformation == "uppercase":
            return str(value).upper() if value is not None else None
        elif transformation == "lowercase":
            return str(value).lower() if value is not None else None
        elif transformation == "strip":
            return str(value).strip() if value is not None else None
        elif transformation == "date":
            if value is None:
                return None
            if isinstance(value, (datetime, date)):
                return value if isinstance(value, date) else value.date()
            if isinstance(value, str):
                # Try common date formats
                for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
                    try:
                        return datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue
                raise ValueError(f"Could not parse date: {value}")
            else:
                raise ValueError(f"Expected string for date conversion, got {type(value)}")
        else:
            # Unknown transformation
            raise ValueError(f"Unknown transformation: {transformation}")