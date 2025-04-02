from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from ...core.database import get_session
from ...models.user import User
from ...models.address import Address
from ...services.address_service import AddressService
from ..dependencies import get_current_active_user

router = APIRouter()

@router.post("/", response_model=Address)
async def create_address(
    address_data: dict,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session)
):
    address_service = AddressService(session)
    validated_data = address_service.validate_address(address_data)
    return address_service.create(validated_data, current_user)

@router.get("/", response_model=List[Address])
async def get_addresses(
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session)
):
    address_service = AddressService(session)
    return address_service.get_all(current_user)

@router.get("/{address_id}", response_model=Address)
async def get_address(
    address_id: int,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session)
):
    address_service = AddressService(session)
    return address_service.get_by_id(address_id, current_user)

@router.put("/{address_id}", response_model=Address)
async def update_address(
    address_id: int,
    address_data: dict,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session)
):
    address_service = AddressService(session)
    validated_data = address_service.validate_address(address_data)
    return address_service.update(address_id, validated_data, current_user)

@router.delete("/{address_id}")
async def delete_address(
    address_id: int,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session)
):
    address_service = AddressService(session)
    address_service.delete(address_id, current_user)
    return {"message": "Address deleted successfully"} 