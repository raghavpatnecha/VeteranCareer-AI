from pydantic import BaseModel, EmailStr, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ServiceTypeEnum(str, Enum):
    """Enumeration for service types."""
    MILITARY = "military"
    GOVERNMENT = "government"
    PARAMILITARY = "paramilitary"


class WorkTypeEnum(str, Enum):
    """Enumeration for work types."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    REMOTE = "remote"
    FREELANCE = "freelance"


class JobAlertFrequencyEnum(str, Enum):
    """Enumeration for job alert frequencies."""
    IMMEDIATE = "immediate"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class GenderEnum(str, Enum):
    """Enumeration for gender."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


# Military ranks by service branch
MILITARY_RANKS = {
    "Army": [
        "Field Marshal", "General", "Lieutenant General", "Major General", "Brigadier",
        "Colonel", "Lieutenant Colonel", "Major", "Captain", "Lieutenant", "Second Lieutenant",
        "Honorary Captain", "Honorary Lieutenant", "Subedar Major", "Subedar", "Naib Subedar",
        "Havildar", "Naik", "Lance Naik", "Sepoy"
    ],
    "Navy": [
        "Admiral of the Fleet", "Admiral", "Vice Admiral", "Rear Admiral", "Commodore",
        "Captain", "Commander", "Lieutenant Commander", "Lieutenant", "Sub Lieutenant",
        "Acting Sub Lieutenant", "Master Chief Petty Officer", "Chief Petty Officer",
        "Petty Officer", "Leading Seaman", "Seaman First Class", "Seaman Second Class"
    ],
    "Air Force": [
        "Marshal of the Air Force", "Air Chief Marshal", "Air Marshal", "Air Vice Marshal",
        "Air Commodore", "Group Captain", "Wing Commander", "Squadron Leader",
        "Flight Lieutenant", "Flying Officer", "Pilot Officer", "Master Warrant Officer",
        "Warrant Officer", "Junior Warrant Officer", "Sergeant", "Corporal",
        "Leading Aircraftman", "Aircraftman"
    ]
}

# Government service branches
GOVERNMENT_SERVICES = [
    "IAS", "IPS", "IFS", "IRS", "Indian Railway Service", "Defence Service",
    "Central Police Service", "State Civil Service", "Municipal Service",
    "Banking Service", "Postal Service", "Income Tax Service", "Customs Service",
    "Forest Service", "Health Service", "Education Service", "Administrative Service"
]

# Paramilitary forces
PARAMILITARY_FORCES = [
    "BSF", "CRPF", "CISF", "ITBP", "SSB", "NSG", "SPG", "RAF",
    "State Police", "Railway Protection Force", "Airport Security Force"
]

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh", "Dadra and Nagar Haveli",
    "Daman and Diu", "Lakshadweep", "Puducherry", "Andaman and Nicobar Islands", "Ladakh", "Jammu and Kashmir"
]


class UserRegistrationSchema(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str
    confirm_password: str
    first_name: str
    last_name: str
    phone: Optional[str] = None
    service_type: ServiceTypeEnum
    service_branch: Optional[str] = None
    rank: Optional[str] = None
    years_of_service: Optional[int] = None
    current_location: Optional[str] = None
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 128:
            raise ValueError('Password must be no more than 128 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*(),.?":{}|<>' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        if v is None:
            return v
        # Remove spaces and hyphens
        phone_clean = v.replace(' ', '').replace('-', '')
        # Indian phone number validation
        if not (phone_clean.isdigit() and 
                (len(phone_clean) == 10 and phone_clean[0] in '6789') or
                (len(phone_clean) == 12 and phone_clean.startswith('91') and phone_clean[2] in '6789') or
                (len(phone_clean) == 13 and phone_clean.startswith('+91') and phone_clean[3] in '6789')):
            raise ValueError('Invalid Indian phone number format')
        return phone_clean
    
    @field_validator('years_of_service')
    @classmethod
    def validate_years_of_service(cls, v):
        if v is not None and (v < 0 or v > 50):
            raise ValueError('Years of service must be between 0 and 50')
        return v
    
    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_names(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters long')
        if len(v.strip()) > 50:
            raise ValueError('Name must be no more than 50 characters long')
        return v.strip().title()
    
    @field_validator('service_branch')
    @classmethod
    def validate_service_branch(cls, v, info):
        if v is None:
            return v
        
        service_type = info.data.get('service_type')
        if service_type == ServiceTypeEnum.MILITARY:
            valid_branches = list(MILITARY_RANKS.keys())
        elif service_type == ServiceTypeEnum.GOVERNMENT:
            valid_branches = GOVERNMENT_SERVICES
        elif service_type == ServiceTypeEnum.PARAMILITARY:
            valid_branches = PARAMILITARY_FORCES
        else:
            return v
        
        if v not in valid_branches:
            raise ValueError(f'Invalid service branch for {service_type}')
        return v
    
    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v, info):
        if v is None:
            return v
        
        service_type = info.data.get('service_type')
        service_branch = info.data.get('service_branch')
        
        if service_type == ServiceTypeEnum.MILITARY and service_branch:
            valid_ranks = MILITARY_RANKS.get(service_branch, [])
            if valid_ranks and v not in valid_ranks:
                raise ValueError(f'Invalid rank for {service_branch}')
        
        return v
    
    @model_validator(mode='after')
    def validate_passwords_match(self):
        password = self.password
        confirm_password = self.confirm_password
        if password and confirm_password and password != confirm_password:
            raise ValueError('Passwords do not match')
        return self


class UserLoginSchema(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str
    remember_me: Optional[bool] = False


class UserProfileUpdateSchema(BaseModel):
    """Schema for user profile updates."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[GenderEnum] = None
    service_branch: Optional[str] = None
    rank: Optional[str] = None
    years_of_service: Optional[int] = None
    retirement_date: Optional[datetime] = None
    last_posting: Optional[str] = None
    military_skills: Optional[List[str]] = None
    civilian_skills: Optional[List[str]] = None
    technical_skills: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    education_level: Optional[str] = None
    specializations: Optional[List[str]] = None
    current_location: Optional[str] = None
    preferred_locations: Optional[List[str]] = None
    willing_to_relocate: Optional[bool] = None
    preferred_work_type: Optional[WorkTypeEnum] = None
    preferred_industries: Optional[List[str]] = None
    preferred_job_titles: Optional[List[str]] = None
    expected_salary_min: Optional[int] = None
    expected_salary_max: Optional[int] = None
    notice_period_days: Optional[int] = None
    email_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    job_alert_frequency: Optional[JobAlertFrequencyEnum] = None
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        if v is None:
            return v
        phone_clean = v.replace(' ', '').replace('-', '')
        if not (phone_clean.isdigit() and 
                (len(phone_clean) == 10 and phone_clean[0] in '6789') or
                (len(phone_clean) == 12 and phone_clean.startswith('91') and phone_clean[2] in '6789') or
                (len(phone_clean) == 13 and phone_clean.startswith('+91') and phone_clean[3] in '6789')):
            raise ValueError('Invalid Indian phone number format')
        return phone_clean
    
    @field_validator('years_of_service')
    @classmethod
    def validate_years_of_service(cls, v):
        if v is not None and (v < 0 or v > 50):
            raise ValueError('Years of service must be between 0 and 50')
        return v
    
    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_names(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Name cannot be empty')
            if len(v.strip()) < 2:
                raise ValueError('Name must be at least 2 characters long')
            if len(v.strip()) > 50:
                raise ValueError('Name must be no more than 50 characters long')
            return v.strip().title()
        return v
    
    @field_validator('expected_salary_min', 'expected_salary_max')
    @classmethod
    def validate_salary(cls, v):
        if v is not None and (v < 0 or v > 100000000):  # Max 10 crores
            raise ValueError('Salary must be between 0 and 10 crores')
        return v
    
    @field_validator('notice_period_days')
    @classmethod
    def validate_notice_period(cls, v):
        if v is not None and (v < 0 or v > 365):
            raise ValueError('Notice period must be between 0 and 365 days')
        return v
    
    @field_validator('military_skills', 'civilian_skills', 'technical_skills', 'certifications', 'specializations', 'preferred_locations', 'preferred_industries', 'preferred_job_titles')
    @classmethod
    def validate_lists(cls, v):
        if v is not None:
            if len(v) > 50:
                raise ValueError('List cannot contain more than 50 items')
            return [item.strip() for item in v if item.strip()]
        return v


class PasswordChangeSchema(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str
    confirm_new_password: str
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 128:
            raise ValueError('Password must be no more than 128 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*(),.?":{}|<>' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @model_validator(mode='after')
    def validate_passwords_match(self):
        new_password = self.new_password
        confirm_new_password = self.confirm_new_password
        if new_password and confirm_new_password and new_password != confirm_new_password:
            raise ValueError('New passwords do not match')
        return self


class PasswordResetRequestSchema(BaseModel):
    """Schema for password reset request."""
    email: EmailStr


class PasswordResetSchema(BaseModel):
    """Schema for password reset."""
    token: str
    new_password: str
    confirm_new_password: str
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 128:
            raise ValueError('Password must be no more than 128 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*(),.?":{}|<>' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @model_validator(mode='after')
    def validate_passwords_match(self):
        new_password = self.new_password
        confirm_new_password = self.confirm_new_password
        if new_password and confirm_new_password and new_password != confirm_new_password:
            raise ValueError('Passwords do not match')
        return self


class UserResponseSchema(BaseModel):
    """Schema for user response data."""
    id: int
    user_id: str
    email: str
    first_name: str
    last_name: str
    full_name: str
    phone: Optional[str] = None
    service_type: str
    service_branch: Optional[str] = None
    rank: Optional[str] = None
    years_of_service: Optional[int] = None
    service_display: str
    current_location: Optional[str] = None
    profile_completion_score: float
    is_active: bool
    is_verified: bool
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    
    class Config:
        from_attributes = True


class UserProfileResponseSchema(BaseModel):
    """Schema for detailed user profile response."""
    id: int
    user_id: str
    email: str
    phone: Optional[str] = None
    first_name: str
    last_name: str
    full_name: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    service_type: str
    service_branch: Optional[str] = None
    rank: Optional[str] = None
    years_of_service: Optional[int] = None
    retirement_date: Optional[str] = None
    last_posting: Optional[str] = None
    service_display: str
    total_experience_years: int
    military_skills: List[str]
    civilian_skills: List[str]
    technical_skills: List[str]
    certifications: List[str]
    education_level: Optional[str] = None
    specializations: List[str]
    current_location: Optional[str] = None
    preferred_locations: List[str]
    willing_to_relocate: bool
    preferred_work_type: str
    preferred_industries: List[str]
    preferred_job_titles: List[str]
    expected_salary_min: Optional[int] = None
    expected_salary_max: Optional[int] = None
    notice_period_days: int
    profile_completion_score: float
    resume_uploaded: bool
    email_notifications: bool
    sms_notifications: bool
    job_alert_frequency: str
    is_active: bool
    is_verified: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_login: Optional[str] = None
    
    class Config:
        from_attributes = True


class AuthTokenSchema(BaseModel):
    """Schema for authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    user: UserResponseSchema


class TokenRefreshSchema(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str


class EmailVerificationSchema(BaseModel):
    """Schema for email verification."""
    token: str


class UserStatsSchema(BaseModel):
    """Schema for user statistics."""
    total_applications: int
    active_applications: int
    interviews_scheduled: int
    offers_received: int
    profile_views: int
    job_matches_count: int
    average_match_score: float
    
    class Config:
        from_attributes = True


class ServiceBranchesResponseSchema(BaseModel):
    """Schema for service branches response."""
    military: Dict[str, List[str]]
    government: List[str]
    paramilitary: List[str]


class RanksResponseSchema(BaseModel):
    """Schema for ranks by service branch response."""
    service_branch: str
    ranks: List[str]