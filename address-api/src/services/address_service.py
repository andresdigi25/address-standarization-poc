from typing import List, Optional
from sqlmodel import Session, select
from ..models.address import Address, AddressCreate, AddressRead
from ..models.user import User
from ..core.exceptions import AddressNotFoundError, AddressValidationError

class AddressService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, address_data: AddressCreate, user: User) -> Address:
        validated_data = self.validate_address(address_data.dict())
        address = Address(**validated_data, user_id=user.id)
        self.session.add(address)
        self.session.commit()
        self.session.refresh(address)
        return address

    def get_by_id(self, address_id: int, user: User) -> Optional[Address]:
        statement = select(Address).where(
            Address.id == address_id,
            Address.user_id == user.id
        )
        address = self.session.exec(statement).first()
        if not address:
            raise AddressNotFoundError(str(address_id))
        return address

    def get_all(self, user: User) -> List[Address]:
        statement = select(Address).where(Address.user_id == user.id)
        return self.session.exec(statement).all()

    def update(self, address_id: int, address_data: AddressCreate, user: User) -> Address:
        address = self.get_by_id(address_id, user)
        validated_data = self.validate_address(address_data.dict())
        for key, value in validated_data.items():
            setattr(address, key, value)
        self.session.commit()
        self.session.refresh(address)
        return address

    def delete(self, address_id: int, user: User) -> None:
        address = self.get_by_id(address_id, user)
        self.session.delete(address)
        self.session.commit()

    def validate_address(self, address_data: dict) -> dict:
        required_fields = ["street", "city", "state", "postal_code", "country"]
        for field in required_fields:
            if not address_data.get(field):
                raise AddressValidationError(f"Missing required field: {field}")
        return address_data 