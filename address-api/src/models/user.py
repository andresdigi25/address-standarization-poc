from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from .address import Address

class UserBase(SQLModel):
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)

class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    addresses: List["Address"] = Relationship(back_populates="user")

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id: int 