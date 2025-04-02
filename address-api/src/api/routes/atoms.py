from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from ...core.database import get_session
from ...models.atom import Atom, AtomCreate, AtomRead
from ...services.atom_service import AtomService

router = APIRouter()

@router.post("/", response_model=AtomRead)
def create_atom(atom_data: AtomCreate, session: Session = Depends(get_session)):
    atom_service = AtomService(session)
    return atom_service.create(atom_data)

@router.get("/", response_model=List[AtomRead])
def get_atoms(session: Session = Depends(get_session)):
    atom_service = AtomService(session)
    return atom_service.get_all()

@router.get("/{atom_id}", response_model=AtomRead)
def get_atom(atom_id: int, session: Session = Depends(get_session)):
    atom_service = AtomService(session)
    atom = atom_service.get_by_id(atom_id)
    if not atom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with id {atom_id} not found"
        )
    return atom

@router.get("/symbol/{symbol}", response_model=AtomRead)
def get_atom_by_symbol(symbol: str, session: Session = Depends(get_session)):
    atom_service = AtomService(session)
    atom = atom_service.get_by_symbol(symbol)
    if not atom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with symbol {symbol} not found"
        )
    return atom

@router.put("/{atom_id}", response_model=AtomRead)
def update_atom(atom_id: int, atom_data: AtomCreate, session: Session = Depends(get_session)):
    atom_service = AtomService(session)
    atom = atom_service.update(atom_id, atom_data)
    if not atom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with id {atom_id} not found"
        )
    return atom

@router.delete("/{atom_id}")
def delete_atom(atom_id: int, session: Session = Depends(get_session)):
    atom_service = AtomService(session)
    if not atom_service.delete(atom_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with id {atom_id} not found"
        )
    return {"message": "Atom deleted successfully"} 