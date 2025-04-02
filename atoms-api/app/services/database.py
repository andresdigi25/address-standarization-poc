import os
from sqlmodel import SQLModel, Session, create_engine

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/atom_db")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)


def create_db_and_tables():
    """Create database tables on application startup"""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Database session dependency"""
    with Session(engine) as session:
        print(f"Database URL: {DATABASE_URL}")
        yield session