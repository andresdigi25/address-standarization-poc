from datetime import date
from typing import Optional
from sqlmodel import SQLModel, Field

class AtomBase(SQLModel):
    name: str = Field(index=True)
    symbol: str
    atomic_number: int
    atomic_mass: float
    discovery_date: Optional[date] = None

class Atom(AtomBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class AtomCreate(AtomBase):
    pass

class AtomRead(AtomBase):
    id: int 