
import requests

# Make a GET request to the API endpoint
response = requests.get("http://localhost:8000/geocode", params={"address": "Stillwell Avenue, Coney Island, Brooklyn, Kings County, City of New York, New York, 11224, United States"})

# Print the response
print(response.json())