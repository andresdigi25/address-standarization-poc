from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class AddressAPIException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class AddressValidationError(AddressAPIException):
    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class AddressNotFoundError(AddressAPIException):
    def __init__(self, address_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Address with id {address_id} not found"
        )

class DatabaseError(AddressAPIException):
    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {detail}"
        )

# Atom-specific exceptions
class AtomValidationError(AddressAPIException):
    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class AtomNotFoundError(AddressAPIException):
    def __init__(self, atom_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with id {atom_id} not found"
        )

class AtomSymbolNotFoundError(AddressAPIException):
    def __init__(self, symbol: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Atom with symbol {symbol} not found"
        )

# User-specific exceptions
class UserValidationError(AddressAPIException):
    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class UserNotFoundError(AddressAPIException):
    def __init__(self, user_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found"
        )

class UserAlreadyExistsError(AddressAPIException):
    def __init__(self, field: str, value: str) -> None:
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with {field} '{value}' already exists"
        )

class InvalidCredentialsError(AddressAPIException):
    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        ) 