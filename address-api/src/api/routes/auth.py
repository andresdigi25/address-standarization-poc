from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from ...core.config import get_settings
from ...core.database import get_session
from ...services.user_service import UserService
from ..dependencies import create_access_token, get_current_user
from ...models.user import User

router = APIRouter()
settings = get_settings()

@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    user_service = UserService(session)
    user = user_service.authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
async def register(
    username: str,
    email: str,
    password: str,
    session: Session = Depends(get_session)
):
    user_service = UserService(session)
    if user_service.get_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    if user_service.get_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    user = user_service.create(username=username, email=email, password=password)
    return {"message": "User created successfully"}

@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user 