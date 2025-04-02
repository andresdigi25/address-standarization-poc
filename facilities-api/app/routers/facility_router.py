from fastapi import APIRouter, HTTPException
from typing import List
from models.facility_model import FacilityRecord  # Updated import
from services.facility_service import FacilityService

facility_router = APIRouter()

facility_service = FacilityService()

@facility_router.post("/facilities/", response_model=FacilityRecord)
def create_facility(facility: FacilityRecord):
    try:
        return facility_service.create_facility(facility)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@facility_router.get("/facilities/", response_model=List[FacilityRecord])
def read_facilities():
    try:
        return facility_service.read_facilities()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@facility_router.get("/facilities/{facility_id}", response_model=FacilityRecord)
def read_facility(facility_id: int):
    try:
        return facility_service.read_facility(facility_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@facility_router.put("/facilities/{facility_id}", response_model=FacilityRecord)
def update_facility(facility_id: int, updated_facility: FacilityRecord):
    try:
        return facility_service.update_facility(facility_id, updated_facility)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@facility_router.delete("/facilities/{facility_id}")
def delete_facility(facility_id: int):
    try:
        return facility_service.delete_facility(facility_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
