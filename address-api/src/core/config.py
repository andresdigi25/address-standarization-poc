from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Address Standardization API"
    
    # Database Settings
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str
    POSTGRES_PORT: str
    POSTGRES_DB: str
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 