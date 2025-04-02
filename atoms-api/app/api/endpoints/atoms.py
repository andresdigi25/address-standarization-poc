from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select
from pydantic import BaseModel

from app.models.atom import Atom
from app.services.database import get_session

router = APIRouter()


class AtomsResponse(BaseModel):
    total: int
    atoms: List[Atom]


@router.get("/atoms", response_model=AtomsResponse)
def get_atoms(
    offset: int = 0,
    limit: int = Query(default=100, le=1000),
    session: Session = Depends(get_session)
):
    """
    Get all atoms from the database with pagination support
    
    - offset: Number of records to skip
    - limit: Maximum number of records to return (max 1000)
    """
    # Debugging - print the table name and column names
    print(f"Table name: {Atom.__tablename__}")
    print(f"Columns: {Atom.__table__.columns.keys()}")
    
    # Debugging - count total records
    count_stmt = select(Atom)
    total_count = len(session.exec(count_stmt).all())
    print(f"Total records in database: {total_count}")
    
    # Original query with pagination
    statement = select(Atom).offset(offset).limit(limit)
    atoms = session.exec(statement).all()
    
    # Debugging - print number of records returned
    print(f"Records returned: {len(atoms)}")
    if atoms:
        print(f"First record ID: {atoms[0].atom_id}")
    
    return AtomsResponse(total=total_count, atoms=atoms)


@router.get("/atoms/{atom_id}", response_model=Atom)
def get_atom_by_id(
    atom_id: int,
    session: Session = Depends(get_session)
):
    """
    Get a specific atom by ID
    
    - atom_id: Primary key of the atom record
    """
    atom = session.get(Atom, atom_id)
    if not atom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with ID {atom_id} not found"
        )
    
    return atom