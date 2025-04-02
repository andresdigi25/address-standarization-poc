import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    USE_POSTGRES: bool = True
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "facilitydb"

    @property
    def DATABASE_URL(self) -> str:
        if self.USE_POSTGRES:
            return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        # Use mounted data directory for SQLite database
        data_dir = "/app/data"
        os.makedirs(data_dir, exist_ok=True)
        sqlite_path = os.path.join(data_dir, "facilities.db")
        return f"sqlite:///{sqlite_path}"

    class Config:
        env_file = ".env"
