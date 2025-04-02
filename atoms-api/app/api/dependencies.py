from typing import Dict, Annotated, Generator
from fastapi import Depends, HTTPException, status
from sqlmodel import Session

from app.services.database import get_session
from app.schemas.mapping import FileMapping, StoredMappings


# Re-export the session dependency for convenience
DatabaseSession = Annotated[Session, Depends(get_session)]


def get_stored_mappings() -> StoredMappings:
    """
    Dependency to access stored mappings
    
    In a real application, this would likely fetch from a database
    rather than using a global variable.
    """
    from app.api.endpoints.upload import stored_mappings
    return stored_mappings


def get_mapping_by_name(
    name: str,
    stored_mappings: StoredMappings = Depends(get_stored_mappings)
) -> FileMapping:
    """
    Dependency to get a specific mapping by name
    
    Args:
        name: Name of the mapping to retrieve
        stored_mappings: StoredMappings dependency
        
    Returns:
        The requested FileMapping
        
    Raises:
        HTTPException: If mapping not found
    """
    if name not in stored_mappings.mappings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mapping '{name}' not found"
        )
    
    return stored_mappings.mappings[name]