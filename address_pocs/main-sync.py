# main.py
import os
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Session, create_engine, select
from sqlalchemy import func
from geopy.geocoders import Nominatim

from models import Outlet
from schemas import OutletCreate

# Read the database URL from the environment (provided by docker-compose)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database.db")
engine = create_engine(DATABASE_URL, echo=True)

app = FastAPI()

# Create the database tables on startup
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

# Dependency for getting a session
def get_session():
    with Session(engine) as session:
        yield session

# Set up a Nominatim geolocator
geolocator = Nominatim(user_agent="outlet_api")

def standardize_address(address1: str, city: str):
    """
    Given an address1 and city, query Nominatim to retrieve the standardized address
    and geolocation (latitude and longitude). Returns a tuple: (formatted_address, lat, lon).
    """
    full_address = f"{address1}, {city}"
    location = geolocator.geocode(full_address)
    if location:
        return location.address, location.latitude, location.longitude
    return None, None, None

@app.post("/outlets", response_model=Outlet)
def create_outlet(outlet: OutletCreate, session: Session = Depends(get_session)):
    # Use Nominatim to standardize the address and get coordinates
    standardized_address, lat, lon = standardize_address(outlet.address1, outlet.city)
    if not lat or not lon:
        raise HTTPException(status_code=404, detail="Address not found")

    # Simple deduplication logic: look for an existing outlet with similar coordinates
    tolerance = 0.001  # degrees tolerance
    statement = select(Outlet).where(
        func.abs(Outlet.latitude - lat) < tolerance,
        func.abs(Outlet.longitude - lon) < tolerance
    )
    existing_outlet = session.exec(statement).first()
    if existing_outlet:
        return existing_outlet

    # If not found, create a new outlet record
    new_outlet = Outlet(
        name=outlet.name,
        address1=outlet.address1,
        address2=outlet.address2,
        city=outlet.city,
        latitude=lat,
        longitude=lon
    )
    session.add(new_outlet)
    session.commit()
    session.refresh(new_outlet)
    return new_outlet

@app.get("/outlets", response_model=list[Outlet])
def read_outlets(session: Session = Depends(get_session)):
    outlets = session.exec(select(Outlet)).all()
    return outlets
