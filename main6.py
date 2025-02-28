from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from geopy.geocoders import Nominatim

app = FastAPI()
geolocator = Nominatim(user_agent="address_standardizer")

class AddressRequest(BaseModel):
    address: str
    city: str
    zipCode: str

class AddressResponse(BaseModel):
    standardizedAddress: str
    latitude: float
    longitude: float

@app.post("/standardize-address", response_model=AddressResponse)
async def standardize_address(request: AddressRequest):
    location = geolocator.geocode(f"{request.address}, {request.city}, {request.zipCode}, USA")

    if location:
        return AddressResponse(
            standardizedAddress=location.address,
            latitude=location.latitude,
            longitude=location.longitude
        )
    else:
        raise HTTPException(status_code=404, detail="Address not found")

# ...existing code...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
