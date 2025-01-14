from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from geopy.geocoders import Nominatim
import folium

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

@app.get("/map", response_class=HTMLResponse)
async def get_map(address: str, city: str, zipCode: str):
    location = geolocator.geocode(f"{address}, {city}, {zipCode}, USA")

    if location:
        m = folium.Map(location=[location.latitude, location.longitude], zoom_start=15)
        folium.Marker([location.latitude, location.longitude], popup=location.address).add_to(m)
        map_html = m._repr_html_()
        return HTMLResponse(content=map_html)
    else:
        raise HTTPException(status_code=404, detail="Address not found")
    
# ...existing code...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
