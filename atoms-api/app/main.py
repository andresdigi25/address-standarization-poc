from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import upload, atoms
from app.services.database import create_db_and_tables

# Create FastAPI app
app = FastAPI(
    title="Atom Data API",
    description="API for uploading, validating, and mapping CSV and JSON files to the Atom model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["File Upload"])
app.include_router(atoms.router, prefix="/api", tags=["Atoms"])

# Startup event
@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Atom Data API is running"}