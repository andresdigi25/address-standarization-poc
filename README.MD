# Address Standardization POC

## Setup

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
```

### Using curl
```bash
curl -X POST "http://127.0.0.1:8000/standardize-address" -H "Content-Type: application/json" -d '{"address": "1600 Amphitheatre Parkway", "city": "Mountain View", "zipCode": "94043"}'
```