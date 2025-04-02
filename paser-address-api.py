from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from address_parser import extract_address_components
from nltk_parser import extract_address_nltk
from address_standardizers import extract_address_usaddress
from typing import Dict

app = FastAPI(title="Address Parser API")

class AddressRequest(BaseModel):
    address: str

class AddressResponse(BaseModel):
    street: str | None
    number: str | None
    city: str | None
    state: str | None
    zip_code: str | None

class ComparisonResponse(BaseModel):
    regex: AddressResponse
    nltk: AddressResponse
    usaddress: AddressResponse

@app.post("/parse-address", response_model=AddressResponse)
async def parse_address(request: AddressRequest):
    try:
        components = extract_address_components(request.address)
        return AddressResponse(**components)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/parse-address-nltk", response_model=AddressResponse)
async def parse_address_nltk(request: AddressRequest):
    try:
        components = extract_address_nltk(request.address)
        return AddressResponse(**components)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/parse-address-usaddress", response_model=AddressResponse)
async def parse_address_usaddress(request: AddressRequest):
    try:
        components = extract_address_usaddress(request.address)
        return AddressResponse(**components)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/compare-parsers", response_model=ComparisonResponse)
async def compare_parsers(request: AddressRequest):
    try:
        return ComparisonResponse(
            regex=AddressResponse(**extract_address_components(request.address)),
            nltk=AddressResponse(**extract_address_nltk(request.address)),
            usaddress=AddressResponse(**extract_address_usaddress(request.address))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Address Parser API is running"}
