# Address Standardization POC

## Setup.

Install the required dependencies:
```bash
pip install fastapi uvicorn geopy
```

## Running the Server

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

## Testing the Endpoint

### Payload
```json
{
  "address": "1600 Amphitheatre Parkway",
  "city": "Mountain View",
  "zipCode": "94043"
}

{
  "address": "STILLWELL AVENUE, BROOKLYN",
  "city": "NY",
  "zipCode": "2313"
}

{
  "address": "2313 STILLWELL AV",
  "city": "",
  "zipCode": ""
}

{
  "address": "2313 STILLWELL AV, BROOKLYN",
  "city": "",
  "zipCode": ""
}
```

### Using curl
```bash
curl -X POST "http://127.0.0.1:8000/standardize-address" -H "Content-Type: application/json" -d '{"address": "1600 Amphitheatre Parkway", "city": "Mountain View", "zipCode": "94043"}'
```

### test for api main2: http://localhost:8000/map?address=STILLWELL%20AVENUE%2C%20BROOKLYN&city=NY&zipCode=2313