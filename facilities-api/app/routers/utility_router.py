from fastapi import APIRouter
from sqlalchemy.sql import text
from db import engine

utility_router = APIRouter()

@utility_router.get("/database-info/")
def get_database_info():
    """
    Returns the type of database being used (PostgreSQL or SQLite) and its version.
    """
    database_type = "PostgreSQL" if "postgresql" in str(engine.url) else "SQLite"
    try:
        with engine.connect() as connection:
            if "postgresql" in str(engine.url):
                version = connection.execute(text("SELECT version();")).scalar()
            else:
                version = connection.execute(text("SELECT sqlite_version();")).scalar()
    except Exception as e:
        version = f"Error retrieving version: {str(e)}"
    
    return {"database_type": database_type, "database_version": version}

@utility_router.get("/healthcheck/")
def health_check():
    """
    Returns the health status of the API.
    """
    return {"status": "healthy"}
