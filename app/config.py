from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    # Application Settings
    app_name: str = "VeteranCareer AI"
    app_version: str = "1.0.0"
    debug: bool = True
    environment: str = "development"
    
    # Database Configuration
    database_url: str = "postgresql://username:password@localhost:5432/veterancareer_db"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "veterancareer_db"
    db_user: str = "username"
    db_password: str = "password"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # JWT Configuration
    jwt_secret_key: str = "your-super-secret-jwt-key-change-this-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS Settings
    allowed_hosts: List[str] = Field(default=["*.clackypaas.com", "localhost", "127.0.0.1"], alias="ALLOWED_HOSTS")
    cors_origins: List[str] = ["http://localhost:3000", "https://*.clackypaas.com"]
    
    # Celery Configuration
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Web Scraping Configuration
    scraping_delay: int = 2
    max_concurrent_requests: int = 5
    user_agent: str = "VeteranCareer-Bot/1.0"
    
    # Job Portal API Keys
    naukri_api_key: Optional[str] = None
    linkedin_api_key: Optional[str] = None
    indeed_api_key: Optional[str] = None
    
    # Email Configuration
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: str = "noreply@veterancareer.ai"
    
    # Machine Learning Model Settings
    ml_model_path: str = "./models/"
    skill_matching_threshold: float = 0.7
    job_matching_threshold: int = 60
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Security Settings
    rate_limit_per_minute: int = 100
    
    # Third-party Services
    openai_api_key: Optional[str] = None
    google_maps_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()