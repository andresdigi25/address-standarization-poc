from fastapi import FastAPI, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import date
import os

class Atom(SQLModel, table=True):
    atom_id: int | None = Field(default=None, primary_key=True)
    source: str
    facility_name: str
    addr1: str
    addr2: str | None = None
    city: str
    state: str
    zip: str
    auth_type: str
    auth_id: str
    expire_date: date | None = None
    first_observed: date | None = None
    data_type: str
    class_of_trade: str

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/atoms")

engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/atoms/")
def create_atom(atom: Atom):
    with Session(engine) as session:
        session.add(atom)
        session.commit()
        session.refresh(atom)
        return atom

@app.get("/atoms/")
def read_atoms():
    with Session(engine) as session:
        atoms = session.exec(select(Atom)).all()
        return atoms

@app.get("/atoms/{atom_id}")
def read_atom(atom_id: int):
    with Session(engine) as session:
        atom = session.get(Atom, atom_id)
        if not atom:
            raise HTTPException(status_code=404, detail="Atom not found")
        return atom

@app.delete("/atoms/{atom_id}")
def delete_atom(atom_id: int):
    with Session(engine) as session:
        atom = session.get(Atom, atom_id)
        if not atom:
            raise HTTPException(status_code=404, detail="Atom not found")
        session.delete(atom)
        session.commit()
        return {"ok": True}

@app.put("/atoms/{atom_id}")
def update_atom(atom_id: int, updated_atom: Atom):
    with Session(engine) as session:
        atom = session.get(Atom, atom_id)
        if not atom:
            raise HTTPException(status_code=404, detail="Atom not found")
        for key, value in updated_atom.dict(exclude_unset=True).items():
            setattr(atom, key, value)
        session.add(atom)
        session.commit()
        session.refresh(atom)
        return atom