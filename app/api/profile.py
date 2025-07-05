from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

from app.database import get_db
from app.models.user import User
from app.models.application import Application
from app.models.job import Job
from app.auth.authentication import get_current_user, get_current_active_user
from app.schemas.user import (
    UserProfileUpdateSchema, 
    UserProfileResponseSchema,
    UserResponseSchema,
    UserStatsSchema
)
from app.ml.skill_translator import MilitarySkillTranslator
from app.ml.career_advisor import CareerAdvisor
from app.tasks.matching_tasks import calculate_user_job_matches, generate_user_recommendations
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/me", response_model=UserProfileResponseSchema)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's detailed profile information.
    """
    try:
        # Update profile completion score
        current_user.calculate_profile_completion()
        db.commit()
        
        return UserProfileResponseSchema(**current_user.to_dict())
        
    except Exception as e:
        logger.error(f"Error retrieving user profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving profile"
        )


@router.put("/me", response_model=UserProfileResponseSchema)
async def update_user_profile(
    profile_data: UserProfileUpdateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current user's profile information.
    """
    try:
        # Update only provided fields
        update_data = profile_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(current_user, field) and value is not None:
                setattr(current_user, field, value)
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        # Save changes
        db.commit()
        db.refresh(current_user)
        
        return UserProfileResponseSchema(**current_user.to_dict())
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating profile for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.get("/me/skills", response_model=Dict[str, Any])
async def get_user_skills(
    current_user: User = Depends(get_current_user)
):
    """
    Get user's skills categorized by type with analysis.
    """
    try:
        skills_data = {
            "military_skills": current_user.military_skills or [],
            "civilian_skills": current_user.civilian_skills or [],
            "technical_skills": current_user.technical_skills or [],
            "certifications": current_user.certifications or [],
            "specializations": current_user.specializations or []
        }
        
        # Calculate skill statistics
        total_skills = sum(len(skills) for skills in skills_data.values())
        
        skill_analysis = {
            "total_skills": total_skills,
            "skills_by_category": {
                category: len(skills) 
                for category, skills in skills_data.items()
            },
            "skill_strength_indicators": {
                "leadership_skills": len([s for s in (current_user.military_skills or []) 
                                        if any(keyword in s.lower() for keyword in ['lead', 'manage', 'command', 'supervise'])]),
                "technical_skills": len(current_user.technical_skills or []),
                "certification_count": len(current_user.certifications or [])
            }
        }
        
        return {
            "skills": skills_data,
            "analysis": skill_analysis,
            "recommendations": {
                "add_civilian_equivalents": len(current_user.military_skills or []) > len(current_user.civilian_skills or []),
                "get_certifications": len(current_user.certifications or []) < 3,
                "technical_upskilling": len(current_user.technical_skills or []) < 5
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving skills for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving skills"
        )


@router.put("/me/skills", response_model=Dict[str, Any])
async def update_user_skills(
    military_skills: Optional[List[str]] = None,
    civilian_skills: Optional[List[str]] = None,
    technical_skills: Optional[List[str]] = None,
    certifications: Optional[List[str]] = None,
    specializations: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's skills with automatic military-to-civilian translation.
    """
    try:
        skill_translator = MilitarySkillTranslator()
        updated_skills = {}
        
        # Update military skills and get civilian translations
        if military_skills is not None:
            current_user.military_skills = military_skills
            updated_skills["military_skills"] = military_skills
            
            # Auto-translate military skills to civilian equivalents
            if military_skills:
                translations = skill_translator.translate_multiple_skills(military_skills)
                suggested_civilian_skills = []
                
                for translation in translations:
                    if translation.confidence_score >= 0.7:
                        suggested_civilian_skills.append(translation.civilian_equivalent)
                
                updated_skills["suggested_civilian_skills"] = suggested_civilian_skills
        
        # Update other skill categories
        if civilian_skills is not None:
            current_user.civilian_skills = civilian_skills
            updated_skills["civilian_skills"] = civilian_skills
        
        if technical_skills is not None:
            current_user.technical_skills = technical_skills
            updated_skills["technical_skills"] = technical_skills
        
        if certifications is not None:
            current_user.certifications = certifications
            updated_skills["certifications"] = certifications
        
        if specializations is not None:
            current_user.specializations = specializations
            updated_skills["specializations"] = specializations
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        # Save changes
        db.commit()
        db.refresh(current_user)
        
        # Trigger background job matching recalculation
        try:
            calculate_user_job_matches.delay(current_user.id)
        except Exception as e:
            logger.warning(f"Failed to trigger job matching update: {str(e)}")
        
        return {
            "message": "Skills updated successfully",
            "updated_skills": updated_skills,
            "profile_completion_score": current_user.profile_completion_score,
            "next_steps": [
                "Review suggested civilian skill translations",
                "Consider adding relevant certifications",
                "Update job preferences based on new skills"
            ]
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating skills for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Skills update failed"
        )


@router.get("/me/preferences", response_model=Dict[str, Any])
async def get_user_preferences(
    current_user: User = Depends(get_current_user)
):
    """
    Get user's job search preferences and settings.
    """
    try:
        preferences = {
            "job_preferences": {
                "preferred_locations": current_user.preferred_locations or [],
                "willing_to_relocate": current_user.willing_to_relocate,
                "preferred_work_type": current_user.preferred_work_type,
                "preferred_industries": current_user.preferred_industries or [],
                "preferred_job_titles": current_user.preferred_job_titles or []
            },
            "salary_expectations": {
                "expected_salary_min": current_user.expected_salary_min,
                "expected_salary_max": current_user.expected_salary_max,
                "notice_period_days": current_user.notice_period_days
            },
            "notification_preferences": {
                "email_notifications": current_user.email_notifications,
                "sms_notifications": current_user.sms_notifications,
                "job_alert_frequency": current_user.job_alert_frequency
            },
            "profile_settings": {
                "profile_completion_score": current_user.profile_completion_score,
                "resume_uploaded": current_user.resume_uploaded,
                "profile_visibility": "active" if current_user.is_active else "inactive"
            }
        }
        
        return {
            "preferences": preferences,
            "recommendations": {
                "complete_salary_range": not (current_user.expected_salary_min and current_user.expected_salary_max),
                "add_more_locations": len(current_user.preferred_locations or []) < 3,
                "specify_industries": len(current_user.preferred_industries or []) < 2,
                "enable_notifications": not current_user.email_notifications
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving preferences for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving preferences"
        )


@router.put("/me/preferences", response_model=Dict[str, Any])
async def update_user_preferences(
    preferred_locations: Optional[List[str]] = None,
    willing_to_relocate: Optional[bool] = None,
    preferred_work_type: Optional[str] = None,
    preferred_industries: Optional[List[str]] = None,
    preferred_job_titles: Optional[List[str]] = None,
    expected_salary_min: Optional[int] = None,
    expected_salary_max: Optional[int] = None,
    notice_period_days: Optional[int] = None,
    email_notifications: Optional[bool] = None,
    sms_notifications: Optional[bool] = None,
    job_alert_frequency: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's job search preferences and notification settings.
    """
    try:
        updated_preferences = {}
        
        # Update job preferences
        if preferred_locations is not None:
            current_user.preferred_locations = preferred_locations
            updated_preferences["preferred_locations"] = preferred_locations
        
        if willing_to_relocate is not None:
            current_user.willing_to_relocate = willing_to_relocate
            updated_preferences["willing_to_relocate"] = willing_to_relocate
        
        if preferred_work_type is not None:
            current_user.preferred_work_type = preferred_work_type
            updated_preferences["preferred_work_type"] = preferred_work_type
        
        if preferred_industries is not None:
            current_user.preferred_industries = preferred_industries
            updated_preferences["preferred_industries"] = preferred_industries
        
        if preferred_job_titles is not None:
            current_user.preferred_job_titles = preferred_job_titles
            updated_preferences["preferred_job_titles"] = preferred_job_titles
        
        # Update salary expectations
        if expected_salary_min is not None:
            current_user.expected_salary_min = expected_salary_min
            updated_preferences["expected_salary_min"] = expected_salary_min
        
        if expected_salary_max is not None:
            current_user.expected_salary_max = expected_salary_max
            updated_preferences["expected_salary_max"] = expected_salary_max
        
        if notice_period_days is not None:
            current_user.notice_period_days = notice_period_days
            updated_preferences["notice_period_days"] = notice_period_days
        
        # Update notification preferences
        if email_notifications is not None:
            current_user.email_notifications = email_notifications
            updated_preferences["email_notifications"] = email_notifications
        
        if sms_notifications is not None:
            current_user.sms_notifications = sms_notifications
            updated_preferences["sms_notifications"] = sms_notifications
        
        if job_alert_frequency is not None:
            current_user.job_alert_frequency = job_alert_frequency
            updated_preferences["job_alert_frequency"] = job_alert_frequency
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        # Save changes
        db.commit()
        db.refresh(current_user)
        
        # Trigger background job matching recalculation
        try:
            calculate_user_job_matches.delay(current_user.id)
        except Exception as e:
            logger.warning(f"Failed to trigger job matching update: {str(e)}")
        
        return {
            "message": "Preferences updated successfully",
            "updated_preferences": updated_preferences,
            "profile_completion_score": current_user.profile_completion_score,
            "impact_analysis": {
                "job_matching_updated": True,
                "notification_settings_changed": any(key in updated_preferences for key in ["email_notifications", "sms_notifications", "job_alert_frequency"]),
                "search_criteria_broadened": "willing_to_relocate" in updated_preferences and updated_preferences["willing_to_relocate"]
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating preferences for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preferences update failed"
        )


@router.get("/me/experience", response_model=Dict[str, Any])
async def get_user_experience(
    current_user: User = Depends(get_current_user)
):
    """
    Get user's experience information and analysis.
    """
    try:
        experience_data = {
            "service_information": {
                "service_type": current_user.service_type,
                "service_branch": current_user.service_branch,
                "rank": current_user.rank,
                "years_of_service": current_user.years_of_service,
                "retirement_date": current_user.retirement_date.isoformat() if current_user.retirement_date else None,
                "last_posting": current_user.last_posting,
                "service_display": current_user.get_service_display()
            },
            "professional_experience": {
                "total_experience_years": current_user.total_experience_years,
                "education_level": current_user.education_level,
                "specializations": current_user.specializations or []
            },
            "career_transition": {
                "profile_completion": current_user.profile_completion_score,
                "civilian_skills_mapped": len(current_user.civilian_skills or []),
                "military_skills_count": len(current_user.military_skills or [])
            }
        }
        
        # Generate experience analysis
        analysis = {
            "leadership_experience": current_user.rank and any(keyword in current_user.rank.lower() for keyword in ['officer', 'commander', 'lead']),
            "technical_background": current_user.service_branch and any(keyword in current_user.service_branch.lower() for keyword in ['engineer', 'technical', 'signals']),
            "management_potential": current_user.years_of_service and current_user.years_of_service >= 10,
            "transition_readiness": current_user.profile_completion_score >= 70
        }
        
        return {
            "experience": experience_data,
            "analysis": analysis,
            "recommendations": {
                "highlight_leadership": analysis["leadership_experience"],
                "emphasize_technical_skills": analysis["technical_background"],
                "target_management_roles": analysis["management_potential"],
                "complete_profile": not analysis["transition_readiness"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving experience for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving experience"
        )


@router.put("/me/experience", response_model=Dict[str, Any])
async def update_user_experience(
    service_branch: Optional[str] = None,
    rank: Optional[str] = None,
    years_of_service: Optional[int] = None,
    retirement_date: Optional[datetime] = None,
    last_posting: Optional[str] = None,
    education_level: Optional[str] = None,
    specializations: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user's experience and service information.
    """
    try:
        updated_fields = {}
        
        # Update service information
        if service_branch is not None:
            current_user.service_branch = service_branch
            updated_fields["service_branch"] = service_branch
        
        if rank is not None:
            current_user.rank = rank
            updated_fields["rank"] = rank
        
        if years_of_service is not None:
            current_user.years_of_service = years_of_service
            # Update total experience if not set
            if not current_user.total_experience_years:
                current_user.total_experience_years = years_of_service
            updated_fields["years_of_service"] = years_of_service
        
        if retirement_date is not None:
            current_user.retirement_date = retirement_date
            updated_fields["retirement_date"] = retirement_date.isoformat()
        
        if last_posting is not None:
            current_user.last_posting = last_posting
            updated_fields["last_posting"] = last_posting
        
        if education_level is not None:
            current_user.education_level = education_level
            updated_fields["education_level"] = education_level
        
        if specializations is not None:
            current_user.specializations = specializations
            updated_fields["specializations"] = specializations
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        # Save changes
        db.commit()
        db.refresh(current_user)
        
        return {
            "message": "Experience updated successfully",
            "updated_fields": updated_fields,
            "profile_completion_score": current_user.profile_completion_score,
            "service_display": current_user.get_service_display(),
            "recommendations": {
                "update_skills": "Consider updating military skills based on new service information",
                "review_job_preferences": "Review job preferences to match updated experience level"
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating experience for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Experience update failed"
        )


@router.get("/me/analytics", response_model=Dict[str, Any])
async def get_profile_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive profile analytics and insights.
    """
    try:
        # Basic profile metrics
        profile_metrics = {
            "profile_completion_score": current_user.profile_completion_score,
            "profile_strength": "strong" if current_user.profile_completion_score >= 80 else "good" if current_user.profile_completion_score >= 60 else "needs_improvement",
            "last_updated": current_user.updated_at.isoformat() if current_user.updated_at else None,
            "account_age_days": (datetime.now() - current_user.created_at).days if current_user.created_at else 0
        }
        
        # Application statistics
        applications = db.query(Application).filter(Application.user_id == current_user.id).all()
        
        application_stats = {
            "total_applications": len(applications),
            "active_applications": len([app for app in applications if app.is_active]),
            "applications_this_month": len([app for app in applications if app.applied_date and app.applied_date >= datetime.now().replace(day=1)]),
            "average_response_time": calculate_average_response_time(applications),
            "success_rate": calculate_application_success_rate(applications)
        }
        
        # Job matching insights
        matching_insights = await get_job_matching_insights(current_user, db)
        
        # Skill analysis
        skill_analysis = {
            "total_skills": len((current_user.military_skills or []) + (current_user.civilian_skills or []) + (current_user.technical_skills or [])),
            "skill_categories": {
                "military": len(current_user.military_skills or []),
                "civilian": len(current_user.civilian_skills or []),
                "technical": len(current_user.technical_skills or []),
                "certifications": len(current_user.certifications or [])
            },
            "skill_strength_score": calculate_skill_strength_score(current_user)
        }
        
        # Market insights
        market_insights = {
            "profile_competitiveness": calculate_profile_competitiveness(current_user, db),
            "industry_alignment": calculate_industry_alignment(current_user),
            "salary_market_position": calculate_salary_market_position(current_user, db)
        }
        
        # Recommendations
        recommendations = generate_profile_recommendations(current_user, application_stats, skill_analysis)
        
        return {
            "profile_metrics": profile_metrics,
            "application_statistics": application_stats,
            "job_matching_insights": matching_insights,
            "skill_analysis": skill_analysis,
            "market_insights": market_insights,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating analytics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating analytics"
        )


@router.get("/me/career-advice", response_model=Dict[str, Any])
async def get_career_advice(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get personalized career advice and recommendations.
    """
    try:
        # Generate comprehensive career advice
        career_advisor = CareerAdvisor()
        
        user_profile_dict = {
            "user_id": current_user.id,
            "military_skills": current_user.military_skills or [],
            "civilian_skills": current_user.civilian_skills or [],
            "technical_skills": current_user.technical_skills or [],
            "years_of_service": current_user.years_of_service or 0,
            "service_branch": current_user.service_branch or '',
            "rank": current_user.rank or '',
            "current_location": current_user.current_location or '',
            "preferred_industries": current_user.preferred_industries or [],
            "expected_salary_min": current_user.expected_salary_min,
            "expected_salary_max": current_user.expected_salary_max,
            "education_level": current_user.education_level
        }
        
        career_advice = career_advisor.generate_career_advice(user_profile_dict)
        
        # Trigger background generation of detailed recommendations
        try:
            generate_user_recommendations.delay(current_user.id, ["career_advice", "industry_trends"])
        except Exception as e:
            logger.warning(f"Failed to trigger detailed recommendations: {str(e)}")
        
        return {
            "career_advice": {
                "recommended_industries": career_advice.recommended_industries[:5],
                "career_paths": career_advice.career_paths[:3],
                "skill_gaps": career_advice.skill_gaps[:10],
                "salary_insights": career_advice.salary_insights,
                "next_steps": career_advice.next_steps[:5],
                "timeline_recommendations": career_advice.timeline_recommendations[:3]
            },
            "confidence_score": career_advice.confidence_score,
            "market_trends": career_advice.market_trends,
            "personalization_note": f"Based on your {current_user.service_branch or 'military'} background and {current_user.years_of_service or 0} years of service",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating career advice for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating career advice"
        )


@router.post("/me/upload-resume")
async def upload_resume(
    resume: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process user's resume.
    """
    try:
        # Validate file type
        allowed_types = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        if resume.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF and Word documents are allowed"
            )
        
        # Validate file size (max 5MB)
        resume_content = await resume.read()
        if len(resume_content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size must be less than 5MB"
            )
        
        # TODO: Implement actual file storage and processing
        # For now, just mark as uploaded
        current_user.resume_uploaded = True
        current_user.resume_file_path = f"resumes/{current_user.id}_{resume.filename}"
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        db.commit()
        
        return {
            "message": "Resume uploaded successfully",
            "filename": resume.filename,
            "file_size": len(resume_content),
            "content_type": resume.content_type,
            "profile_completion_score": current_user.profile_completion_score,
            "next_steps": [
                "Resume will be processed for skill extraction",
                "Check back for automated skill suggestions",
                "Review and update your profile based on resume analysis"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading resume for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resume upload failed"
        )


@router.delete("/me/resume")
async def delete_resume(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete user's uploaded resume.
    """
    try:
        if not current_user.resume_uploaded:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No resume found to delete"
            )
        
        # TODO: Implement actual file deletion from storage
        
        # Update user record
        current_user.resume_uploaded = False
        current_user.resume_file_path = None
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        db.commit()
        
        return {
            "message": "Resume deleted successfully",
            "profile_completion_score": current_user.profile_completion_score
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting resume for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resume deletion failed"
        )


@router.get("/me/recommendations/refresh")
async def refresh_recommendations(
    current_user: User = Depends(get_current_user)
):
    """
    Trigger refresh of job recommendations and career advice.
    """
    try:
        # Trigger background tasks for recommendation refresh
        job_matching_task = calculate_user_job_matches.delay(current_user.id)
        recommendations_task = generate_user_recommendations.delay(
            current_user.id, 
            ["jobs", "skills", "career_advice", "industry_trends"]
        )
        
        return {
            "message": "Recommendation refresh initiated",
            "tasks": {
                "job_matching": job_matching_task.id,
                "recommendations": recommendations_task.id
            },
            "estimated_completion": "2-3 minutes",
            "next_steps": [
                "Check back in a few minutes for updated recommendations",
                "Visit the jobs section to see new matches",
                "Review updated career advice"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error refreshing recommendations for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh recommendations"
        )


# Helper functions

def calculate_average_response_time(applications: List[Application]) -> Optional[float]:
    """Calculate average response time for applications."""
    response_times = []
    for app in applications:
        if app.applied_date and app.company_response_date:
            delta = app.company_response_date - app.applied_date
            response_times.append(delta.days)
    
    return sum(response_times) / len(response_times) if response_times else None


def calculate_application_success_rate(applications: List[Application]) -> float:
    """Calculate application success rate."""
    if not applications:
        return 0.0
    
    successful_apps = len([app for app in applications if app.status in ['offered', 'accepted']])
    return (successful_apps / len(applications)) * 100


async def get_job_matching_insights(user: User, db: Session) -> Dict[str, Any]:
    """Get job matching insights for the user."""
    # Get recent job matches (this would typically come from cached results)
    recent_jobs = db.query(Job).filter(
        Job.is_active == True,
        Job.veteran_match_score >= 60
    ).limit(10).all()
    
    if not recent_jobs:
        return {
            "average_match_score": 0.0,
            "total_matches": 0,
            "top_industries": [],
            "matching_quality": "insufficient_data",
            "last_updated": datetime.now().isoformat()
        }
    
    # Calculate matching insights from recent jobs
    match_scores = [job.veteran_match_score for job in recent_jobs if job.veteran_match_score]
    industries = [job.industry for job in recent_jobs if job.industry]
    
    return {
        "average_match_score": sum(match_scores) / len(match_scores) if match_scores else 0.0,
        "total_matches": len(recent_jobs),
        "top_industries": list(set(industries))[:5],
        "matching_quality": "excellent" if sum(match_scores) / len(match_scores) >= 80 else "good" if sum(match_scores) / len(match_scores) >= 60 else "fair",
        "last_updated": datetime.now().isoformat()
    }