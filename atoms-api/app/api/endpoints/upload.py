import json
import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlmodel import Session, select
from pydantic import BaseModel
from datetime import datetime, date
from enum import Enum

from app.schemas.mapping import FileMapping, StoredMappings
from app.services.database import get_session
from app.services.file_parser import FileParser
from app.services.validator import DataValidator
from app.models.atom import Atom


class MappingNameEnum(str, Enum):
    standard_csv = "standard_csv"
    alternate_csv = "alternate_csv"
    standard_json = "standard_json"
    alternate_json = "alternate_json"


router = APIRouter()

# In-memory storage for mappings
# In a production app, these would be stored in the database
stored_mappings = StoredMappings()


class ValidationError(BaseModel):
    field: str
    message: str


class UploadResponse(BaseModel):
    filename: str
    mapping_used: str
    total_records: int
    successful_records: int
    failed_records: int
    parse_errors: List[Dict[str, Any]] = []
    validation_errors: List[Dict[str, Any]] = []

    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }


# Add some default mappings
default_mappings = {
    "standard_csv": FileMapping(
        name="standard_csv",
        description="Standard CSV mapping",
        file_type="csv",
        columns=[
            {"source": "Facility", "target": "facility_name", "required": True},
            {"source": "Address1", "target": "addr1", "required": True},
            {"source": "Address2", "target": "addr2", "required": False},
            {"source": "City", "target": "city", "required": True},
            {"source": "State", "target": "state", "required": True},
            {"source": "ZIP", "target": "zip", "required": True},
            {"source": "Authorization Type", "target": "auth_type", "required": True},
            {"source": "Authorization ID", "target": "auth_id", "required": True},
            {"source": "Expiration Date", "target": "expire_date", "required": False, "transformation": "date"},
            {"source": "First Observed", "target": "first_observed", "required": False, "transformation": "date"},
            {"source": "Data Type", "target": "data_type", "required": True},
            {"source": "Class of Trade", "target": "class_of_trade", "required": True}
        ]
    ),
    "alternate_csv": FileMapping(
        name="alternate_csv",
        description="Alternate CSV mapping with different column names",
        file_type="csv",
        columns=[
            {"source": "FacilityName", "target": "facility_name", "required": True},
            {"source": "AddressLine1", "target": "addr1", "required": True},
            {"source": "AddressLine2", "target": "addr2", "required": False},
            {"source": "City", "target": "city", "required": True},
            {"source": "StateCode", "target": "state", "required": True},
            {"source": "PostalCode", "target": "zip", "required": True},
            {"source": "AuthType", "target": "auth_type", "required": True},
            {"source": "AuthID", "target": "auth_id", "required": True},
            {"source": "ExpirationDate", "target": "expire_date", "required": False, "transformation": "date"},
            {"source": "FirstObserved", "target": "first_observed", "required": False, "transformation": "date"},
            {"source": "DataType", "target": "data_type", "required": True},
            {"source": "TradeClass", "target": "class_of_trade", "required": True}
        ]
    ),
    "standard_json": FileMapping(
        name="standard_json",
        description="Standard JSON mapping",
        file_type="json",
        columns=[
            {"source": "facility", "target": "facility_name", "required": True},
            {"source": "address_line_1", "target": "addr1", "required": True},
            {"source": "address_line_2", "target": "addr2", "required": False},
            {"source": "city", "target": "city", "required": True},
            {"source": "state", "target": "state", "required": True},
            {"source": "zip_code", "target": "zip", "required": True},
            {"source": "authorization_type", "target": "auth_type", "required": True},
            {"source": "authorization_id", "target": "auth_id", "required": True},
            {"source": "expiration_date", "target": "expire_date", "required": False, "transformation": "date"},
            {"source": "first_observed", "target": "first_observed", "required": False, "transformation": "date"},
            {"source": "data_type", "target": "data_type", "required": True},
            {"source": "class_of_trade", "target": "class_of_trade", "required": True}
        ]
    ),
    "alternate_json": FileMapping(
        name="alternate_json",
        description="Alternate JSON mapping with different field names and stricter validation",
        file_type="json",
        columns=[
            {"source": "FacilityName", "target": "facility_name", "required": True, "transformation": "strip"},
            {"source": "AddressLine1", "target": "addr1", "required": True, "transformation": "strip"},
            {"source": "AddressLine2", "target": "addr2", "required": False, "transformation": "strip"},
            {"source": "City", "target": "city", "required": True, "transformation": "strip"},
            {"source": "StateCode", "target": "state", "required": True, "transformation": "uppercase"},
            {"source": "PostalCode", "target": "zip", "required": True, "transformation": "strip"},
            {"source": "AuthType", "target": "auth_type", "required": True, "transformation": "uppercase"},
            {"source": "AuthID", "target": "auth_id", "required": True, "transformation": "strip"},
            {"source": "ExpirationDate", "target": "expire_date", "required": True, "transformation": "date"},
            {"source": "FirstObserved", "target": "first_observed", "required": True, "transformation": "date"},
            {"source": "DataType", "target": "data_type", "required": True, "transformation": "uppercase"},
            {"source": "TradeClass", "target": "class_of_trade", "required": True, "transformation": "uppercase"}
        ]
    )
}

stored_mappings.mappings.update(default_mappings)


@router.post("/mappings", status_code=status.HTTP_201_CREATED, response_model=FileMapping)
def create_mapping(mapping: FileMapping):
    """Create a new column mapping profile"""
    if mapping.name in stored_mappings.mappings:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Mapping with name '{mapping.name}' already exists"
        )
    
    stored_mappings.mappings[mapping.name] = mapping
    return mapping


@router.get("/mappings", response_model=Dict[str, FileMapping])
def get_mappings():
    """Get all available mapping profiles"""
    return stored_mappings.mappings


@router.get("/mappings/{name}", response_model=FileMapping)
def get_mapping(name: str):
    """Get a specific mapping profile by name"""
    if name not in stored_mappings.mappings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mapping '{name}' not found"
        )
    
    return stored_mappings.mappings[name]


@router.delete("/mappings/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_mapping(name: str):
    """Delete a mapping profile by name"""
    if name not in stored_mappings.mappings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mapping '{name}' not found"
        )
    
    del stored_mappings.mappings[name]
    return None


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    mapping_name: MappingNameEnum = Form(...),
    session: Session = Depends(get_session)
):
    """
    Upload and process a file using a specified mapping profile
    
    - The file will be parsed according to the specified mapping
    - Records will be validated against the Atom model
    - Valid records will be stored in the database
    - A summary of the processing will be returned
    """
    # Convert enum to string
    mapping_name_str = mapping_name.value
    
    # Check if mapping exists (should always exist since we're using an enum)
    if mapping_name_str not in stored_mappings.mappings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mapping '{mapping_name_str}' not found"
        )
    
    mapping = stored_mappings.mappings[mapping_name_str]
    
    # Check file type
    file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    expected_extension = ".csv" if mapping.file_type.lower() == "csv" else ".json"
    
    if expected_extension not in file_extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {mapping.file_type} file, got {file_extension}"
        )
    
    # Read file content
    file_content = await file.read()
    
    try:
        # Parse file based on mapping
        valid_records, parse_errors = FileParser.parse_file(file_content, mapping)
        print(f"After parsing: {len(valid_records)} valid records, {len(parse_errors)} errors")
        if parse_errors:
            print("Parse errors details:")
            for error in parse_errors[:3]:  # Show first 3 errors
                print(f"- {error}")
        
        if valid_records:
            print("Sample valid record:")
            class DateEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, date):
                        return obj.isoformat()
                    return super().default(obj)
            print(json.dumps(valid_records[0], indent=2, cls=DateEncoder))
        
        # Validate records against Atom model
        validated_records, validation_errors = DataValidator.validate_records(valid_records)
        print(f"After validation: {len(validated_records)} valid records, {len(validation_errors)} errors")
        if validation_errors:
            print("Validation errors details:")
            for error in validation_errors[:3]:  # Show first 3 errors
                print(f"- {error}")
            
        # If all records failed, add more detailed error information to the response
        detailed_errors = []
        if len(valid_records) > 0 and len(validated_records) == 0:
            for error in validation_errors[:3]:  # Show first 3 errors for brevity
                if "validation_errors" in error:
                    detailed_errors.append(error["validation_errors"])
            print(f"Detailed validation errors (first 3): {detailed_errors}")
        
        # Create Atom instances
        atom_instances = DataValidator.create_atom_instances(validated_records)
        print(f"Created {len(atom_instances)} Atom instances")
        
        # Save to database
        print(f"Saving {len(atom_instances)} records to database")
        for atom in atom_instances:
            session.add(atom)
        
        session.commit()
        
        # Verify records were saved by counting
        verification_stmt = select(Atom)
        total_after = len(session.exec(verification_stmt).all())
        print(f"Total records in database after commit: {total_after}")
        
        # Create response using the Pydantic model
        response = UploadResponse(
            filename=file.filename or "",
            mapping_used=mapping_name_str,
            total_records=len(valid_records) + len(parse_errors),
            successful_records=len(validated_records),
            failed_records=len(parse_errors) + len(validation_errors),
            parse_errors=[{k: v.isoformat() if isinstance(v, date) else v for k, v in error.items()} for error in parse_errors],
            validation_errors=[{k: v.isoformat() if isinstance(v, date) else v for k, v in error.items()} for error in validation_errors]
        )
        
        return response
        
    except Exception as e:
        import traceback
        error_detail = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log the full error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/mappings/debug/{name}")
def debug_mapping(name: str):
    """Debug endpoint to display mapping details"""
    if name not in stored_mappings.mappings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mapping '{name}' not found"
        )
    
    mapping = stored_mappings.mappings[name]
    
    # Convert to a simple dictionary for display
    columns_info = []
    for col in mapping.columns:
        columns_info.append({
            "source": col.source,
            "target": col.target,
            "required": col.required,
            "default": col.default,
            "transformation": col.transformation
        })
    
    return {
        "name": mapping.name,
        "description": mapping.description,
        "file_type": mapping.file_type,
        "columns": columns_info
    }