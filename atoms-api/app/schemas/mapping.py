from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class ColumnMapping(BaseModel):
    """Schema for mapping file columns to model fields"""
    source: str = Field(..., description="Field name in source file")
    target: str = Field(..., description="Field name in Atom model")
    required: bool = Field(default=True, description="Whether the field is required")
    default: Optional[str] = Field(default=None, description="Default value if field is missing or empty")
    transformation: Optional[str] = Field(default=None, description="Optional transformation to apply (e.g., 'uppercase', 'lowercase', 'strip')")


class FileMapping(BaseModel):
    """Schema for file mapping configuration"""
    name: str = Field(..., description="Name of the mapping profile")
    description: Optional[str] = Field(default=None, description="Description of what this mapping is for")
    file_type: str = Field(..., description="Type of file (csv, json)")
    columns: List[ColumnMapping] = Field(..., description="Column mapping definitions")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "standard_csv_mapping",
                "description": "Standard mapping for facility data files",
                "file_type": "csv",
                "columns": [
                    {
                        "source": "Facility",
                        "target": "facility_name",
                        "required": True
                    },
                    {
                        "source": "Address Line 1",
                        "target": "addr1",
                        "required": True
                    },
                    {
                        "source": "Address Line 2",
                        "target": "addr2",
                        "required": False,
                        "default": ""
                    }
                ]
            }
        }


class StoredMappings(BaseModel):
    """Schema for stored mapping profiles"""
    mappings: Dict[str, FileMapping] = {}