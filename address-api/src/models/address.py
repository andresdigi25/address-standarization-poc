from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship
from .user import User

class AddressBase(SQLModel):
    street: str = Field(index=True)
    city: str = Field(index=True)
    state: str
    postal_code: str
    country: str

class Address(AddressBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: int = Field(foreign_key="user.id")
    user: User = Relationship(back_populates="addresses")

class AddressCreate(AddressBase):
    pass

class AddressRead(AddressBase):
    id: int
    created_at: datetime
    updated_at: datetime
    user_id: int 