from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import get_settings
from core.logging import logger
from core.database import init_db
from api.routes import auth, addresses, atoms

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(addresses.router, prefix=f"{settings.API_V1_STR}/addresses", tags=["addresses"])
app.include_router(atoms.router, prefix=f"{settings.API_V1_STR}/atoms", tags=["atoms"])

@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    return {"message": "Welcome to Address Standardization API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}