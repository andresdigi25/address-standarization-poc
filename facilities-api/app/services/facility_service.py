from sqlmodel import Session, select
from models.facility_model import FacilityRecord
from db import engine  # Updated import
from datetime import date

class FacilityService:
    def create_facility(self, facility: FacilityRecord) -> FacilityRecord:
        if isinstance(facility.expire_date, str):
            facility.expire_date = date.fromisoformat(facility.expire_date)
        if isinstance(facility.first_observed, str):
            facility.first_observed = date.fromisoformat(facility.first_observed)
        with Session(engine) as session:
            session.add(facility)
            session.commit()
            session.refresh(facility)
            return facility

    def read_facilities(self) -> list[FacilityRecord]:
        with Session(engine) as session:
            return session.exec(select(FacilityRecord)).all()

    def read_facility(self, facility_id: int) -> FacilityRecord:
        with Session(engine) as session:
            facility = session.get(FacilityRecord, facility_id)
            if not facility:
                raise ValueError("Facility not found")
            return facility

    def update_facility(self, facility_id: int, updated_facility: FacilityRecord) -> FacilityRecord:
        with Session(engine) as session:
            facility = session.get(FacilityRecord, facility_id)
            if not facility:
                raise ValueError("Facility not found")
            for key, value in updated_facility.dict(exclude_unset=True).items():
                setattr(facility, key, value)
            session.add(facility)
            session.commit()
            session.refresh(facility)
            return facility

    def delete_facility(self, facility_id: int) -> dict:
        with Session(engine) as session:
            facility = session.get(FacilityRecord, facility_id)
            if not facility:
                raise ValueError("Facility not found")
            session.delete(facility)
            session.commit()
            return {"detail": "Facility deleted successfully"}
