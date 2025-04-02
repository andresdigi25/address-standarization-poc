from typing import List, Optional
from sqlmodel import Session, select
from ..models.atom import Atom, AtomCreate, AtomRead
from ..core.exceptions import AtomNotFoundError, AtomSymbolNotFoundError, AtomValidationError

class AtomService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, atom_data: AtomCreate) -> Atom:
        validated_data = self.validate_atom(atom_data.dict())
        atom = Atom(**validated_data)
        self.session.add(atom)
        self.session.commit()
        self.session.refresh(atom)
        return atom

    def get_by_id(self, atom_id: int) -> Optional[Atom]:
        statement = select(Atom).where(Atom.id == atom_id)
        atom = self.session.exec(statement).first()
        if not atom:
            raise AtomNotFoundError(str(atom_id))
        return atom

    def get_all(self) -> List[Atom]:
        statement = select(Atom)
        return self.session.exec(statement).all()

    def get_by_symbol(self, symbol: str) -> Optional[Atom]:
        statement = select(Atom).where(Atom.symbol == symbol)
        atom = self.session.exec(statement).first()
        if not atom:
            raise AtomSymbolNotFoundError(symbol)
        return atom

    def update(self, atom_id: int, atom_data: AtomCreate) -> Optional[Atom]:
        atom = self.get_by_id(atom_id)
        validated_data = self.validate_atom(atom_data.dict())
        for key, value in validated_data.items():
            setattr(atom, key, value)
        self.session.commit()
        self.session.refresh(atom)
        return atom

    def delete(self, atom_id: int) -> bool:
        atom = self.get_by_id(atom_id)
        self.session.delete(atom)
        self.session.commit()
        return True

    def validate_atom(self, atom_data: dict) -> dict:
        required_fields = ["name", "symbol", "atomic_number", "atomic_mass"]
        for field in required_fields:
            if not atom_data.get(field):
                raise AtomValidationError(f"Missing required field: {field}")
        
        # Validate atomic number
        if not isinstance(atom_data.get("atomic_number"), int) or atom_data["atomic_number"] <= 0:
            raise AtomValidationError("Atomic number must be a positive integer")
        
        # Validate atomic mass
        if not isinstance(atom_data.get("atomic_mass"), (int, float)) or atom_data["atomic_mass"] <= 0:
            raise AtomValidationError("Atomic mass must be a positive number")
        
        # Validate symbol
        if not isinstance(atom_data.get("symbol"), str) or len(atom_data["symbol"]) > 2:
            raise AtomValidationError("Symbol must be a string of maximum 2 characters")
        
        return atom_data 