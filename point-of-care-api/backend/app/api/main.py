# Install the required packages
# pip install fastapi uvicorn sqlalchemy sqlite pydantic

# main.py
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List
import uvicorn
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

class PointOfCare(Base):
    __tablename__ = "points_of_care"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    address = Column(String)
    phone_number = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PointOfCareCreate(BaseModel):
    name: str
    address: str
    phone_number: str

class PointOfCareResponse(BaseModel):
    id: int
    name: str
    address: str
    phone_number: str

    class Config:
       from_attributes = True

@app.post("/points_of_care/", response_model=PointOfCareResponse)
def create_point_of_care(point_of_care: PointOfCareCreate, db: Session = Depends(get_db)):
    db_point_of_care = PointOfCare(**point_of_care.dict())
    db.add(db_point_of_care)
    db.commit()
    db.refresh(db_point_of_care)
    return db_point_of_care

@app.get("/points_of_care/{point_of_care_id}", response_model=PointOfCareResponse)
def read_point_of_care(point_of_care_id: int, db: Session = Depends(get_db)):
    point_of_care = db.query(PointOfCare).filter(PointOfCare.id == point_of_care_id).first()
    if point_of_care is None:
        raise HTTPException(status_code=404, detail="Point of care not found")
    return point_of_care

@app.put("/points_of_care/{point_of_care_id}", response_model=PointOfCareResponse)
def update_point_of_care(point_of_care_id: int, updated_point_of_care: PointOfCareCreate, db: Session = Depends(get_db)):
    point_of_care = db.query(PointOfCare).filter(PointOfCare.id == point_of_care_id).first()
    if point_of_care is None:
        raise HTTPException(status_code=404, detail="Point of care not found")
    point_of_care.name = updated_point_of_care.name
    point_of_care.address = updated_point_of_care.address
    point_of_care.phone_number = updated_point_of_care.phone_number
    db.commit()
    db.refresh(point_of_care)
    return point_of_care

@app.delete("/points_of_care/{point_of_care_id}")
def delete_point_of_care(point_of_care_id: int, db: Session = Depends(get_db)):
    point_of_care = db.query(PointOfCare).filter(PointOfCare.id == point_of_care_id).first()
    if point_of_care is None:
        raise HTTPException(status_code=404, detail="Point of care not found")
    db.delete(point_of_care)
    db.commit()
    return {"detail": "Point of care deleted"}

@app.get("/points_of_care/", response_model=List[PointOfCareResponse])
def read_points_of_care(db: Session = Depends(get_db)):
    points_of_care = db.query(PointOfCare).all()
    return points_of_care


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,workers =4,reload=False)


# To run the application, use the following command:
# uvicorn main:app --reload