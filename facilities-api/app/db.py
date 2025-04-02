from sqlmodel import create_engine, SQLModel
from config import Settings

settings = Settings()
DATABASE_URL = settings.DATABASE_URL
engine = create_engine(DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)
