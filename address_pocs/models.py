# models.py
from typing import Optional
from sqlmodel import SQLModel, Field

class Outlet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    address1: str
    address2: Optional[str] = None
    city: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
