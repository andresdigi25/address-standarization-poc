import typer
import json
from typing import Optional, List, Any, ClassVar
from datetime import date
from pydantic import BaseModel, Field, field_validator, model_validator
import re
from enum import Enum
import sys

# Define the Facility model with validators
class Facility(BaseModel):
    atom_id: int
    source: str
    facility_name: str
    addr1: str
    addr2: Optional[str] = None
    city: str
    state: str
    zip: str
    auth_type: str
    auth_id: str
    expire_date: date
    first_observed: date
    data_type: str
    class_of_trade: str

    # List of restricted terms for facility_name
    RESTRICTED_TERMS: ClassVar[List[str]] = [
        "BLIND", "BLINDED", "BLOCKED", "BLOCKED PER CUSTOMER", 
        "RGH ENTERPRISE, INC.", "RGH ENTERPRISES", "BYRAM HEALTHCARE", 
        "PHARMACY CONFIDENTIAL LTC", "PHARMACY CONFIDENTIAL ACUTE", 
        "PATIENT DATA", "ZIPPED CUSTOMER NAME", "MPCS BLINDED", 
        "WOUND CARE", "PATIENT SALE", "HOME PATIENT", "MASKED ACUTE", 
        "CONFIDENTIAL", "SHIP TO", "CONFIDENTIAL PATIENT INFO", "XX"
    ]

    # Validator for facility_name to check for restricted terms
    @field_validator('facility_name')
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        # Convert to uppercase for case-insensitive matching
        name_upper = v.upper()
        
        # Check if any restricted term is contained in the facility name
        for term in cls.RESTRICTED_TERMS:
            if term in name_upper:
                raise ValueError(f"Facility name contains restricted term: {term}")
                
        return v
    
    # Validator for addr1 to reject if it starts with 'PATIENT DELIVERY'
    @field_validator('addr1')
    @classmethod
    def validate_addr1(cls, v: str) -> str:
        if v.upper().startswith('PATIENT DELIVERY'):
            raise ValueError("Address cannot start with 'PATIENT DELIVERY'")
        return v
    
    # Validator to ensure required fields are populated
    @field_validator('facility_name', 'addr1', 'city', 'state', 'zip', 'auth_id')
    @classmethod
    def check_not_empty(cls, v: str, info: Any) -> str:
        if not v or v.strip() == '':
            field_name = info.field_name
            raise ValueError(f"Field {field_name} cannot be empty")
        return v

# Initialize the Typer app
app = typer.Typer(help="Facility data validation CLI")

def parse_date(date_str: str) -> date:
    """Convert string date to date object"""
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        typer.echo(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")
        raise typer.Exit(1)

@app.command()
def validate(
    json_file: str = typer.Argument(..., help="Path to JSON file containing facility data"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for valid records")
):
    """Validate facility records from a JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        typer.echo(f"Error: File '{json_file}' not found.")
        raise typer.Exit(1)
    except json.JSONDecodeError:
        typer.echo(f"Error: File '{json_file}' is not a valid JSON file.")
        raise typer.Exit(1)
    
    valid_records = []
    invalid_records = []
    
    # Process each record
    for i, record in enumerate(data):
        try:
            # Convert string dates to date objects
            if 'expire_date' in record and isinstance(record['expire_date'], str):
                record['expire_date'] = parse_date(record['expire_date'])
            if 'first_observed' in record and isinstance(record['first_observed'], str):
                record['first_observed'] = parse_date(record['first_observed'])
                
            # Validate the record
            facility = Facility.model_validate(record)
            valid_records.append(facility.model_dump())
            typer.echo(f"Record {i+1}: Valid")
            
        except Exception as e:
            invalid_records.append({"record_index": i, "error": str(e), "data": record})
            typer.echo(f"Record {i+1}: Invalid - {str(e)}")
    
    # Summary
    typer.echo(f"\nValidation Summary:")
    typer.echo(f"Total records: {len(data)}")
    typer.echo(f"Valid records: {len(valid_records)}")
    typer.echo(f"Invalid records: {len(invalid_records)}")
    
    # Write valid records to output file if specified
    if output_file and valid_records:
        try:
            with open(output_file, 'w') as f:
                json.dump(valid_records, f, indent=2, default=str)
            typer.echo(f"Valid records written to {output_file}")
        except Exception as e:
            typer.echo(f"Error writing to output file: {str(e)}")
    
    # Return exit code 1 if any records were invalid
    if invalid_records:
        raise typer.Exit(1)

@app.command()
def validate_single(
    atom_id: int = typer.Option(..., "--atom-id", help="Facility atom ID"),
    source: str = typer.Option(..., "--source", help="Source of the facility data"),
    facility_name: str = typer.Option(..., "--name", help="Facility name"),
    addr1: str = typer.Option(..., "--addr1", help="Address line 1"),
    addr2: Optional[str] = typer.Option(None, "--addr2", help="Address line 2"),
    city: str = typer.Option(..., "--city", help="City"),
    state: str = typer.Option(..., "--state", help="State"),
    zip_code: str = typer.Option(..., "--zip", help="ZIP code"),
    auth_type: str = typer.Option(..., "--auth-type", help="Authorization type"),
    auth_id: str = typer.Option(..., "--auth-id", help="Authorization ID"),
    expire_date: str = typer.Option(..., "--expire-date", help="Expiration date (YYYY-MM-DD)"),
    first_observed: str = typer.Option(..., "--first-observed", help="First observed date (YYYY-MM-DD)"),
    data_type: str = typer.Option(..., "--data-type", help="Data type"),
    class_of_trade: str = typer.Option(..., "--class", help="Class of trade")
):
    """Validate a single facility record provided via command line arguments."""
    try:
        # Create a facility record
        facility_data = {
            "atom_id": atom_id,
            "source": source,
            "facility_name": facility_name,
            "addr1": addr1,
            "addr2": addr2,
            "city": city,
            "state": state,
            "zip": zip_code,
            "auth_type": auth_type,
            "auth_id": auth_id,
            "expire_date": parse_date(expire_date),
            "first_observed": parse_date(first_observed),
            "data_type": data_type,
            "class_of_trade": class_of_trade
        }
        
        # Validate the facility data
        facility = Facility.model_validate(facility_data)
        
        # If validation passes, output success message
        typer.echo("Validation successful!")
        typer.echo(json.dumps(facility.model_dump(), indent=2, default=str))
        
    except Exception as e:
        typer.echo(f"Validation failed: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()