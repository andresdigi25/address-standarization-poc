from typing import List, Optional
from sqlmodel import Session, select
from ..models.user import User, UserCreate, UserRead
from ..core.auth import get_password_hash, verify_password
from ..core.exceptions import (
    UserNotFoundError,
    UserAlreadyExistsError,
    UserValidationError,
    InvalidCredentialsError
)

class UserService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, user_data: UserCreate) -> User:
        # Check if username exists
        if self.get_by_username(user_data.username):
            raise UserAlreadyExistsError("username", user_data.username)
        
        # Check if email exists
        if self.get_by_email(user_data.email):
            raise UserAlreadyExistsError("email", user_data.email)
        
        validated_data = self.validate_user(user_data.dict())
        hashed_password = get_password_hash(validated_data.pop("password"))
        user = User(hashed_password=hashed_password, **validated_data)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get_by_id(self, user_id: int) -> Optional[User]:
        statement = select(User).where(User.id == user_id)
        user = self.session.exec(statement).first()
        if not user:
            raise UserNotFoundError(str(user_id))
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        statement = select(User).where(User.username == username)
        return self.session.exec(statement).first()

    def get_by_email(self, email: str) -> Optional[User]:
        statement = select(User).where(User.email == email)
        return self.session.exec(statement).first()

    def get_all(self) -> List[User]:
        statement = select(User)
        return self.session.exec(statement).all()

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.get_by_username(username)
        if not user or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError()
        return user

    def update(self, user_id: int, user_data: UserCreate) -> Optional[User]:
        user = self.get_by_id(user_id)
        
        # Check if new username exists (if changed)
        if user_data.username != user.username:
            if self.get_by_username(user_data.username):
                raise UserAlreadyExistsError("username", user_data.username)
        
        # Check if new email exists (if changed)
        if user_data.email != user.email:
            if self.get_by_email(user_data.email):
                raise UserAlreadyExistsError("email", user_data.email)
        
        validated_data = self.validate_user(user_data.dict())
        hashed_password = get_password_hash(validated_data.pop("password"))
        
        for key, value in validated_data.items():
            setattr(user, key, value)
        user.hashed_password = hashed_password
        
        self.session.commit()
        self.session.refresh(user)
        return user

    def delete(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        self.session.delete(user)
        self.session.commit()
        return True

    def validate_user(self, user_data: dict) -> dict:
        required_fields = ["username", "email", "password"]
        for field in required_fields:
            if not user_data.get(field):
                raise UserValidationError(f"Missing required field: {field}")
        
        # Validate email format
        if not "@" in user_data.get("email", ""):
            raise UserValidationError("Invalid email format")
        
        # Validate password strength
        password = user_data.get("password", "")
        if len(password) < 8:
            raise UserValidationError("Password must be at least 8 characters long")
        
        return user_data 