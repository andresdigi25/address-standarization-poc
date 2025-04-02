from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import usaddress
from thefuzz import process

app = FastAPI()

# Mock address database (replace with a real database or API)
REFERENCE_ADDRESSES = [
    "1600 Pennsylvania Ave NW, Washington, DC 20500",
    "350 Fifth Ave, New York, NY 10118",
    "1 Apple Park Way, Cupertino, CA 95014",
    "221B Baker Street, London, UK",  # Non-US for fun!
]

class AddressRequest(BaseModel):
    address: str

class AddressResponse(BaseModel):
    standardized_address: str
    components: dict

class FuzzyMatchResponse(BaseModel):
    input_address: str
    best_match: str
    confidence: float

@app.post("/standardize-address", response_model=AddressResponse)
def standardize_address(request: AddressRequest):
    try:
        # Parse address using usaddress
        parsed_address, address_type = usaddress.tag(request.address)

        # Standardized format
        standardized_address = f"{parsed_address.get('AddressNumber', '')} {parsed_address.get('StreetName', '')} {parsed_address.get('StreetNamePostType', '')}, {parsed_address.get('PlaceName', '')}, {parsed_address.get('StateName', '')} {parsed_address.get('ZipCode', '')}"

        return AddressResponse(
            standardized_address=standardized_address.strip(),
            components=parsed_address
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing address: {str(e)}")

@app.post("/fuzzy-match", response_model=FuzzyMatchResponse)
def fuzzy_match_address(request: AddressRequest):
    best_match, confidence = process.extractOne(request.address, REFERENCE_ADDRESSES)

    if confidence < 70:  # Confidence threshold
        raise HTTPException(status_code=404, detail="No high-confidence match found")

    return FuzzyMatchResponse(
        input_address=request.address,
        best_match=best_match,
        confidence=confidence
    )

# Run with: uvicorn main:app --reload
