from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ApplicationStatus(str, Enum):
    """Enumeration for application status values."""
    DRAFT = "draft"
    APPLIED = "applied"
    VIEWED = "viewed"
    SHORTLISTED = "shortlisted"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    INTERVIEW_COMPLETED = "interview_completed"
    UNDER_REVIEW = "under_review"
    OFFERED = "offered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    ON_HOLD = "on_hold"


class FollowUpType(str, Enum):
    """Enumeration for follow-up reminder types."""
    APPLICATION_REMINDER = "application_reminder"
    INTERVIEW_PREPARATION = "interview_preparation"
    POST_INTERVIEW = "post_interview"
    STATUS_CHECK = "status_check"
    THANK_YOU_NOTE = "thank_you_note"
    SALARY_NEGOTIATION = "salary_negotiation"


class ApplicationCreateSchema(BaseModel):
    """Schema for creating a new job application."""
    model_config = ConfigDict(from_attributes=True)
    
    job_id: int = Field(..., description="ID of the job being applied to")
    status: Optional[ApplicationStatus] = Field(ApplicationStatus.DRAFT, description="Application status")
    application_source: Optional[str] = Field(None, description="Source of application (direct, portal, referral)")
    cover_letter: Optional[str] = Field(None, description="Cover letter content")
    resume_version: Optional[str] = Field(None, description="Version of resume used")
    application_priority: Optional[str] = Field("medium", description="Priority level (high, medium, low)")
    user_notes: Optional[str] = Field(None, description="User's notes about the application")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    hr_contact_name: Optional[str] = Field(None, description="HR contact name")
    hr_contact_email: Optional[str] = Field(None, description="HR contact email")
    hr_contact_phone: Optional[str] = Field(None, description="HR contact phone")
    hiring_manager_name: Optional[str] = Field(None, description="Hiring manager name")
    auto_follow_up_enabled: Optional[bool] = Field(True, description="Enable automatic follow-up reminders")


class ApplicationUpdateSchema(BaseModel):
    """Schema for updating an existing job application."""
    model_config = ConfigDict(from_attributes=True)
    
    status: Optional[ApplicationStatus] = Field(None, description="Updated application status")
    cover_letter: Optional[str] = Field(None, description="Updated cover letter")
    resume_version: Optional[str] = Field(None, description="Updated resume version")
    company_feedback: Optional[str] = Field(None, description="Feedback from company")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection")
    interview_scheduled_date: Optional[datetime] = Field(None, description="Scheduled interview date")
    interview_type: Optional[str] = Field(None, description="Type of interview")
    interview_location: Optional[str] = Field(None, description="Interview location")
    interview_notes: Optional[str] = Field(None, description="Interview notes")
    interview_feedback: Optional[str] = Field(None, description="Interview feedback")
    offered_salary: Optional[int] = Field(None, description="Offered salary amount")
    offer_deadline: Optional[datetime] = Field(None, description="Offer deadline")
    offer_details: Optional[str] = Field(None, description="Offer details")
    negotiation_notes: Optional[str] = Field(None, description="Salary negotiation notes")
    hr_contact_name: Optional[str] = Field(None, description="HR contact name")
    hr_contact_email: Optional[str] = Field(None, description="HR contact email")
    hr_contact_phone: Optional[str] = Field(None, description="HR contact phone")
    hiring_manager_name: Optional[str] = Field(None, description="Hiring manager name")
    application_priority: Optional[str] = Field(None, description="Priority level")
    user_rating: Optional[float] = Field(None, ge=1, le=5, description="User rating (1-5)")
    user_notes: Optional[str] = Field(None, description="User's notes")
    tags: Optional[str] = Field(None, description="Comma-separated tags")
    next_follow_up_date: Optional[datetime] = Field(None, description="Next follow-up date")
    auto_follow_up_enabled: Optional[bool] = Field(None, description="Enable automatic follow-up")
    is_starred: Optional[bool] = Field(None, description="Star important applications")
    is_archived: Optional[bool] = Field(None, description="Archive application")


class ApplicationResponseSchema(BaseModel):
    """Schema for application response data."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    application_id: str
    user_id: int
    job_id: int
    status: ApplicationStatus
    previous_status: Optional[str] = None
    status_updated_at: Optional[datetime] = None
    applied_date: Optional[datetime] = None
    application_source: Optional[str] = None
    cover_letter: Optional[str] = None
    resume_version: Optional[str] = None
    company_response_date: Optional[datetime] = None
    company_feedback: Optional[str] = None
    rejection_reason: Optional[str] = None
    interview_scheduled_date: Optional[datetime] = None
    interview_type: Optional[str] = None
    interview_location: Optional[str] = None
    interview_notes: Optional[str] = None
    interview_feedback: Optional[str] = None
    offer_received_date: Optional[datetime] = None
    offered_salary: Optional[int] = None
    offer_deadline: Optional[datetime] = None
    offer_details: Optional[str] = None
    negotiation_notes: Optional[str] = None
    hr_contact_name: Optional[str] = None
    hr_contact_email: Optional[str] = None
    hr_contact_phone: Optional[str] = None
    hiring_manager_name: Optional[str] = None
    application_priority: str = "medium"
    user_rating: Optional[float] = None
    user_notes: Optional[str] = None
    tags: Optional[str] = None
    match_score_at_application: Optional[float] = None
    success_probability: Optional[float] = None
    next_follow_up_date: Optional[datetime] = None
    follow_up_count: int = 0
    last_follow_up_date: Optional[datetime] = None
    auto_follow_up_enabled: bool = True
    days_since_application: int = 0
    response_time_days: Optional[int] = None
    total_communications: int = 0
    external_application_id: Optional[str] = None
    portal_application_url: Optional[str] = None
    is_active: bool = True
    is_starred: bool = False
    is_archived: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None


class FollowUpReminderSchema(BaseModel):
    """Schema for follow-up reminder creation and response."""
    model_config = ConfigDict(from_attributes=True)
    
    application_id: int = Field(..., description="ID of the associated application")
    reminder_type: FollowUpType = Field(..., description="Type of follow-up reminder")
    reminder_date: datetime = Field(..., description="Date and time for the reminder")
    title: str = Field(..., description="Title of the reminder")
    description: Optional[str] = Field(None, description="Description of the reminder")
    is_completed: Optional[bool] = Field(False, description="Whether the reminder is completed")
    priority: Optional[str] = Field("medium", description="Priority level (high, medium, low)")
    auto_generated: Optional[bool] = Field(False, description="Whether reminder was auto-generated")
    completion_notes: Optional[str] = Field(None, description="Notes when completing the reminder")
