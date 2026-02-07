"""FastAPI application entry point."""
import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import router

app = FastAPI(
    title="SDG System API",
    description="Synthetic Data Generation System API",
    version="1.0.0"
)

def get_allowed_origins() -> List[str]:
    """Get allowed origins from environment."""
    origins_str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173"  # Dev defaults
    )
    return [origin.strip() for origin in origins_str.split(",")]

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
