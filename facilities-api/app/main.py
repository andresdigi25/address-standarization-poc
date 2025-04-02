import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from config import Settings
from routers.facility_router import facility_router
from routers.utility_router import utility_router  # Import the new router
from db import init_db  # Updated import

settings = Settings()

# Configure structured logging
logger = logging.getLogger("uvicorn")
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

app = FastAPI()

# Include the routers
app.include_router(facility_router)
app.include_router(utility_router)  # Add the utility router

# Create database tables
@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Database tables created successfully.")

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace = traceback.format_exc()
    logger.error({
        "error": str(exc),
        "trace": trace,
        "path": request.url.path,
        "method": request.method
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred.", "trace": trace},
    )
