from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI instance
app = FastAPI(
    title="VeteranCareer AI",
    description="AI-powered job discovery and career guidance for Indian ex-servicemen and government officers",
    version="1.0.0"
)

# CORS configuration for Clacky environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "VeteranCareer AI is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "VeteranCareer AI"}

# Include API routers
from app.api import auth, jobs, profile, applications
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(profile.router, prefix="/api/profile", tags=["profile"])
app.include_router(applications.router, prefix="/api/applications", tags=["applications"])