# main.py
import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from models import Outlet
from schemas import OutletCreate

# Read the database URL from the environment (with asyncpg driver)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@db:5432/outlets_db")

# Create an async engine and sessionmaker
engine = create_async_engine(DATABASE_URL, echo=True)
async_session_maker = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()

# Create the database tables on startup
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# Dependency: provide an async session for endpoints
async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

# Set up the Nominatim geolocator.
# Note: geopy is synchronous, so we wrap calls in an executor.
geolocator = Nominatim(user_agent="outlet_api")

async def standardize_address(address1: str, city: str):
    """
    Uses Nominatim to geocode the provided address. Because geopy is synchronous,
    the geocode call is run in an executor.
    Returns a tuple: (formatted_address, latitude, longitude)
    """
    full_address = f"{address1}, {city}"
    loop = asyncio.get_event_loop()
    try:
        location = await loop.run_in_executor(None, geolocator.geocode, full_address)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        raise HTTPException(status_code=503, detail="Geocoding service unavailable") from e

    if location:
        return location.address, location.latitude, location.longitude
    return None, None, None

@app.post("/outlets", response_model=Outlet)
async def create_outlet(outlet: OutletCreate, session: AsyncSession = Depends(get_session)):
    # Standardize the address using the geocoding service
    standardized_address, lat, lon = await standardize_address(outlet.address1, outlet.city)
    if not lat or not lon:
        raise HTTPException(status_code=404, detail="Address not found")

    # Deduplication logic:
    # Look for an existing outlet with coordinates within a small tolerance.
    tolerance = 0.001  # degrees tolerance
    statement = select(Outlet).where(
        func.abs(Outlet.latitude - lat) < tolerance,
        func.abs(Outlet.longitude - lon) < tolerance
    )
    result = await session.execute(statement)
    existing_outlet = result.scalar_one_or_none()
    if existing_outlet:
        return existing_outlet

    # Create and persist a new outlet
    new_outlet = Outlet(
        name=outlet.name,
        address1=outlet.address1,
        address2=outlet.address2,
        city=outlet.city,
        latitude=lat,
        longitude=lon
    )
    session.add(new_outlet)
    try:
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail="Failed to create outlet") from e
    await session.refresh(new_outlet)
    return new_outlet

@app.get("/outlets", response_model=list[Outlet])
async def read_outlets(session: AsyncSession = Depends(get_session)):
    statement = select(Outlet)
    result = await session.execute(statement)
    outlets = result.scalars().all()
    return outlets
