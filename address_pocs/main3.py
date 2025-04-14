from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/geocode")
def geocode(address: str):
    # Make a request to the Nominatim service
    response = requests.get("https://nominatim.openstreetmap.org/search", params={"q": address, "format": "json"})
    data = response.json()
    
    # Extract the latitude and longitude from the response
    lat = data[0]["lat"]
    lon = data[0]["lon"]
    
    return {"latitude": lat, "longitude": lon}

# ...existing code...
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
