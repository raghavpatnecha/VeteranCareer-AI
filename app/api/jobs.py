from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging

from app.database import get_db
from app.models.job import Job
from app.models.user import User
from app.models.application import Application
from app.auth.authentication import get_current_user, get_current_user_optional
from app.ml.job_matcher import JobMatcher, UserProfile
from app.tasks.matching_tasks import calculate_user_job_matches
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def search_jobs(
    # Search parameters
    q: Optional[str] = Query(None, description="Search query for job title, company, or keywords"),
    location: Optional[str] = Query(None, description="Location filter"),
    company: Optional[str] = Query(None, description="Company name filter"),
    
    # Category filters
    industry: Optional[str] = Query(None, description="Industry filter"),
    job_type: Optional[str] = Query(None, description="Job type (full_time, part_time, contract, internship)"),
    experience_level: Optional[str] = Query(None, description="Experience level (entry, mid, senior, executive)"),
    
    # Salary filters
    salary_min: Optional[int] = Query(None, description="Minimum salary filter"),
    salary_max: Optional[int] = Query(None, description="Maximum salary filter"),
    
    # Veteran-specific filters
    veteran_friendly: Optional[bool] = Query(None, description="Filter for veteran-friendly jobs"),
    government_jobs: Optional[bool] = Query(None, description="Filter for government jobs only"),
    psu_jobs: Optional[bool] = Query(None, description="Filter for PSU jobs only"),
    security_clearance: Optional[bool] = Query(None, description="Filter for jobs requiring security clearance"),
    
    # Experience filters
    min_experience: Optional[int] = Query(None, description="Minimum years of experience required"),
    max_experience: Optional[int] = Query(None, description="Maximum years of experience required"),
    
    # Date filters
    posted_since: Optional[int] = Query(None, description="Jobs posted within last N days"),
    
    # Remote work filter
    remote_only: Optional[bool] = Query(None, description="Filter for remote jobs only"),
    
    # Sorting
    sort_by: Optional[str] = Query("relevance", description="Sort by: relevance, date, salary, match_score"),
    sort_order: Optional[str] = Query("desc", description="Sort order: asc, desc"),
    
    # Pagination
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Number of jobs per page"),
    
    # Dependencies
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Search and filter jobs with comprehensive filtering options.
    Returns paginated results with job details and match scores for authenticated users.
    """
    try:
        # Start with base query for active jobs
        query = db.query(Job).filter(Job.is_active == True)
        
        # Apply text search
        if q:
            search_terms = q.lower().split()
            for term in search_terms:
                query = query.filter(
                    or_(
                        Job.title.ilike(f"%{term}%"),
                        Job.company_name.ilike(f"%{term}%"),
                        Job.description.ilike(f"%{term}%"),
                        Job.requirements.ilike(f"%{term}%")
                    )
                )
        
        # Location filter
        if location:
            query = query.filter(Job.location.ilike(f"%{location}%"))
        
        # Company filter
        if company:
            query = query.filter(Job.company_name.ilike(f"%{company}%"))
        
        # Industry filter
        if industry:
            query = query.filter(Job.industry.ilike(f"%{industry}%"))
        
        # Job type filter
        if job_type:
            query = query.filter(Job.job_type == job_type)
        
        # Experience level filter
        if experience_level:
            query = query.filter(Job.experience_level == experience_level)
        
        # Salary filters
        if salary_min:
            query = query.filter(
                or_(
                    Job.salary_min >= salary_min,
                    Job.salary_max >= salary_min
                )
            )
        
        if salary_max:
            query = query.filter(
                or_(
                    Job.salary_max <= salary_max,
                    Job.salary_min <= salary_max
                )
            )
        
        # Veteran-specific filters
        if veteran_friendly:
            query = query.filter(Job.veteran_preference == True)
        
        if government_jobs:
            query = query.filter(Job.government_job == True)
        
        if psu_jobs:
            query = query.filter(Job.psu_job == True)
        
        if security_clearance:
            query = query.filter(Job.security_clearance_required == True)
        
        # Experience filters
        if min_experience is not None:
            query = query.filter(
                or_(
                    Job.min_experience_years <= min_experience,
                    Job.min_experience_years.is_(None)
                )
            )
        
        if max_experience is not None:
            query = query.filter(
                or_(
                    Job.max_experience_years >= max_experience,
                    Job.max_experience_years.is_(None)
                )
            )
        
        # Date filter
        if posted_since:
            cutoff_date = datetime.now() - timedelta(days=posted_since)
            query = query.filter(Job.posted_date >= cutoff_date)
        
        # Remote work filter
        if remote_only:
            query = query.filter(Job.is_remote == True)
        
        # Apply sorting
        if sort_by == "date":
            if sort_order == "asc":
                query = query.order_by(asc(Job.posted_date))
            else:
                query = query.order_by(desc(Job.posted_date))
        elif sort_by == "salary":
            if sort_order == "asc":
                query = query.order_by(asc(Job.salary_max))
            else:
                query = query.order_by(desc(Job.salary_max))
        elif sort_by == "match_score":
            if sort_order == "asc":
                query = query.order_by(asc(Job.veteran_match_score))
            else:
                query = query.order_by(desc(Job.veteran_match_score))
        else:  # relevance - default sorting
            query = query.order_by(desc(Job.veteran_match_score), desc(Job.posted_date))
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * limit
        jobs = query.offset(offset).limit(limit).all()
        
        # Calculate match scores for authenticated users
        job_results = []
        for job in jobs:
            job_dict = job.to_search_result()
            
            # Add personalized match score for authenticated users
            if current_user:
                try:
                    matcher = JobMatcher()
                    user_profile = create_user_profile_from_model(current_user)
                    match_result = matcher.calculate_job_match_score(user_profile, job)
                    
                    job_dict.update({
                        "personal_match_score": round(match_result.match_score, 1),
                        "skill_match": round(match_result.skill_match_score, 1),
                        "experience_match": round(match_result.experience_match_score, 1),
                        "location_match": round(match_result.location_match_score, 1),
                        "match_reasons": match_result.match_reasons[:3]  # Top 3 reasons
                    })
                except Exception as e:
                    logger.warning(f"Error calculating match score for job {job.id}: {str(e)}")
                    job_dict["personal_match_score"] = job.veteran_match_score
            
            job_results.append(job_dict)
        
        # Calculate pagination info
        total_pages = (total_count + limit - 1) // limit
        has_next = page < total_pages
        has_prev = page > 1
        
        return {
            "jobs": job_results,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_count": total_count,
                "has_next": has_next,
                "has_prev": has_prev,
                "limit": limit
            },
            "filters_applied": {
                "search_query": q,
                "location": location,
                "company": company,
                "industry": industry,
                "job_type": job_type,
                "experience_level": experience_level,
                "salary_range": f"{salary_min or 'any'} - {salary_max or 'any'}",
                "veteran_friendly": veteran_friendly,
                "government_jobs": government_jobs,
                "psu_jobs": psu_jobs,
                "remote_only": remote_only,
                "posted_since": f"{posted_since} days" if posted_since else None
            },
            "sort": {
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
    except Exception as e:
        logger.error(f"Error in job search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while searching jobs"
        )


@router.get("/{job_id}", response_model=Dict[str, Any])
async def get_job_details(
    job_id: int = Path(..., description="Job ID"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get detailed information about a specific job.
    Includes personalized match analysis for authenticated users.
    """
    try:
        # Get job details
        job = db.query(Job).filter(Job.id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Get detailed job information
        job_details = job.to_dict(include_sensitive=False)
        
        # Add view tracking
        job.increment_view_count()
        db.commit()
        
        # Add personalized analysis for authenticated users
        if current_user:
            try:
                matcher = JobMatcher()
                user_profile = create_user_profile_from_model(current_user)
                match_result = matcher.calculate_job_match_score(user_profile, job)
                
                job_details.update({
                    "match_analysis": {
                        "overall_score": round(match_result.match_score, 1),
                        "skill_match": round(match_result.skill_match_score, 1),
                        "experience_match": round(match_result.experience_match_score, 1),
                        "location_match": round(match_result.location_match_score, 1),
                        "preference_match": round(match_result.preference_match_score, 1),
                        "salary_match": round(match_result.salary_match_score, 1),
                        "match_reasons": match_result.match_reasons,
                        "improvement_suggestions": match_result.improvement_suggestions
                    }
                })
                
                # Check if user has already applied
                existing_application = db.query(Application).filter(
                    Application.user_id == current_user.id,
                    Application.job_id == job.id
                ).first()
                
                job_details["user_application"] = {
                    "has_applied": existing_application is not None,
                    "application_status": existing_application.status if existing_application else None,
                    "applied_date": existing_application.applied_date.isoformat() if existing_application and existing_application.applied_date else None
                }
                
            except Exception as e:
                logger.warning(f"Error calculating detailed match for job {job_id}: {str(e)}")
        
        # Add similar jobs
        similar_jobs = get_similar_jobs(db, job, limit=5)
        job_details["similar_jobs"] = [job.to_search_result() for job in similar_jobs]
        
        return job_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job details for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving job details"
        )


@router.get("/recommendations/for-me", response_model=Dict[str, Any])
async def get_job_recommendations(
    limit: int = Query(20, ge=1, le=50, description="Number of recommendations"),
    recommendation_type: str = Query("all", description="Type: all, high_match, recent, trending"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get personalized job recommendations for the authenticated user.
    """
    try:
        # Trigger background job matching calculation
        matching_task = calculate_user_job_matches.delay(current_user.id, job_limit=100)
        
        # Get immediate recommendations while background task runs
        recommendations = []
        
        # Base query for user's preferred criteria
        base_query = db.query(Job).filter(
            Job.is_active == True,
            Job.posted_date >= datetime.now() - timedelta(days=60)  # Recent jobs
        )
        
        # Filter based on user preferences
        if current_user.preferred_locations:
            location_filters = []
            for location in current_user.preferred_locations:
                location_filters.append(Job.location.ilike(f"%{location}%"))
            if current_user.willing_to_relocate:
                location_filters.append(Job.is_remote == True)
            base_query = base_query.filter(or_(*location_filters))
        
        if current_user.preferred_industries:
            industry_filters = []
            for industry in current_user.preferred_industries:
                industry_filters.append(Job.industry.ilike(f"%{industry}%"))
            base_query = base_query.filter(or_(*industry_filters))
        
        # Salary filtering
        if current_user.expected_salary_min:
            base_query = base_query.filter(
                or_(
                    Job.salary_max >= current_user.expected_salary_min,
                    Job.salary_max.is_(None)
                )
            )
        
        # Get different types of recommendations
        if recommendation_type == "high_match" or recommendation_type == "all":
            high_match_jobs = base_query.filter(
                Job.veteran_match_score >= 75
            ).order_by(desc(Job.veteran_match_score)).limit(limit // 2).all()
            
            for job in high_match_jobs:
                job_dict = job.to_search_result()
                job_dict["recommendation_reason"] = "High compatibility match"
                job_dict["recommendation_type"] = "high_match"
                recommendations.append(job_dict)
        
        if recommendation_type == "recent" or recommendation_type == "all":
            recent_jobs = base_query.filter(
                Job.posted_date >= datetime.now() - timedelta(days=7),
                Job.veteran_match_score >= 60
            ).order_by(desc(Job.posted_date)).limit(limit // 3).all()
            
            for job in recent_jobs:
                if job.id not in [r["id"] for r in recommendations]:  # Avoid duplicates
                    job_dict = job.to_search_result()
                    job_dict["recommendation_reason"] = "Recently posted"
                    job_dict["recommendation_type"] = "recent"
                    recommendations.append(job_dict)
        
        if recommendation_type == "trending" or recommendation_type == "all":
            trending_jobs = base_query.filter(
                Job.view_count >= 50,  # Popular jobs
                Job.veteran_match_score >= 60
            ).order_by(desc(Job.view_count)).limit(limit // 4).all()
            
            for job in trending_jobs:
                if job.id not in [r["id"] for r in recommendations]:  # Avoid duplicates
                    job_dict = job.to_search_result()
                    job_dict["recommendation_reason"] = "Trending among veterans"
                    job_dict["recommendation_type"] = "trending"
                    recommendations.append(job_dict)
        
        # Add veteran-specific recommendations
        veteran_specific = base_query.filter(
            or_(
                Job.veteran_preference == True,
                Job.government_job == True,
                Job.psu_job == True
            )
        ).order_by(desc(Job.veteran_match_score)).limit(limit // 4).all()
        
        for job in veteran_specific:
            if job.id not in [r["id"] for r in recommendations]:
                job_dict = job.to_search_result()
                job_dict["recommendation_reason"] = "Veteran-friendly employer"
                job_dict["recommendation_type"] = "veteran_specific"
                recommendations.append(job_dict)
        
        # Calculate personal match scores
        matcher = JobMatcher()
        user_profile = create_user_profile_from_model(current_user)
        
        for rec in recommendations:
            try:
                job = db.query(Job).filter(Job.id == rec["id"]).first()
                if job:
                    match_result = matcher.calculate_job_match_score(user_profile, job)
                    rec["personal_match_score"] = round(match_result.match_score, 1)
                    rec["match_breakdown"] = {
                        "skills": round(match_result.skill_match_score, 1),
                        "experience": round(match_result.experience_match_score, 1),
                        "location": round(match_result.location_match_score, 1),
                        "preferences": round(match_result.preference_match_score, 1)
                    }
            except Exception as e:
                logger.warning(f"Error calculating match for recommendation {rec['id']}: {str(e)}")
                rec["personal_match_score"] = rec.get("veteran_match_score", 0)
        
        # Sort by personal match score and limit results
        recommendations.sort(key=lambda x: x.get("personal_match_score", 0), reverse=True)
        recommendations = recommendations[:limit]
        
        return {
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "recommendation_type": recommendation_type,
            "user_preferences": {
                "preferred_locations": current_user.preferred_locations or [],
                "preferred_industries": current_user.preferred_industries or [],
                "willing_to_relocate": current_user.willing_to_relocate,
                "expected_salary_range": {
                    "min": current_user.expected_salary_min,
                    "max": current_user.expected_salary_max
                }
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating job recommendations for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating recommendations"
        )


@router.get("/stats/overview", response_model=Dict[str, Any])
async def get_job_stats_overview(
    db: Session = Depends(get_db)
):
    """
    Get overview statistics about available jobs.
    """
    try:
        # Basic job counts
        total_jobs = db.query(Job).filter(Job.is_active == True).count()
        recent_jobs = db.query(Job).filter(
            Job.is_active == True,
            Job.posted_date >= datetime.now() - timedelta(days=7)
        ).count()
        
        veteran_friendly = db.query(Job).filter(
            Job.is_active == True,
            Job.veteran_preference == True
        ).count()
        
        government_jobs = db.query(Job).filter(
            Job.is_active == True,
            Job.government_job == True
        ).count()
        
        psu_jobs = db.query(Job).filter(
            Job.is_active == True,
            Job.psu_job == True
        ).count()
        
        remote_jobs = db.query(Job).filter(
            Job.is_active == True,
            Job.is_remote == True
        ).count()
        
        # Job type distribution
        job_type_stats = db.query(
            Job.job_type,
            func.count(Job.id).label('count')
        ).filter(Job.is_active == True).group_by(Job.job_type).all()
        
        # Industry distribution
        industry_stats = db.query(
            Job.industry,
            func.count(Job.id).label('count')
        ).filter(
            Job.is_active == True,
            Job.industry.isnot(None)
        ).group_by(Job.industry).order_by(desc(func.count(Job.id))).limit(10).all()
        
        # Location distribution
        location_stats = db.query(
            Job.city,
            func.count(Job.id).label('count')
        ).filter(
            Job.is_active == True,
            Job.city.isnot(None)
        ).group_by(Job.city).order_by(desc(func.count(Job.id))).limit(10).all()
        
        # Experience level distribution
        experience_stats = db.query(
            Job.experience_level,
            func.count(Job.id).label('count')
        ).filter(
            Job.is_active == True,
            Job.experience_level.isnot(None)
        ).group_by(Job.experience_level).all()
        
        # Salary ranges
        salary_ranges = {
            "0-5L": db.query(Job).filter(
                Job.is_active == True,
                Job.salary_max <= 500000
            ).count(),
            "5L-10L": db.query(Job).filter(
                Job.is_active == True,
                Job.salary_min >= 500000,
                Job.salary_max <= 1000000
            ).count(),
            "10L-20L": db.query(Job).filter(
                Job.is_active == True,
                Job.salary_min >= 1000000,
                Job.salary_max <= 2000000
            ).count(),
            "20L+": db.query(Job).filter(
                Job.is_active == True,
                Job.salary_min >= 2000000
            ).count()
        }
        
        return {
            "job_counts": {
                "total_active_jobs": total_jobs,
                "jobs_posted_this_week": recent_jobs,
                "veteran_friendly_jobs": veteran_friendly,
                "government_jobs": government_jobs,
                "psu_jobs": psu_jobs,
                "remote_jobs": remote_jobs
            },
            "distributions": {
                "by_job_type": [{"type": stat.job_type, "count": stat.count} for stat in job_type_stats],
                "by_industry": [{"industry": stat.industry, "count": stat.count} for stat in industry_stats],
                "by_location": [{"location": stat.city, "count": stat.count} for stat in location_stats],
                "by_experience": [{"level": stat.experience_level, "count": stat.count} for stat in experience_stats],
                "by_salary_range": salary_ranges
            },
            "veteran_insights": {
                "veteran_friendly_percentage": round((veteran_friendly / max(total_jobs, 1)) * 100, 1),
                "government_psu_percentage": round(((government_jobs + psu_jobs) / max(total_jobs, 1)) * 100, 1),
                "remote_work_percentage": round((remote_jobs / max(total_jobs, 1)) * 100, 1)
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating job stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating job statistics"
        )


@router.get("/filters/options", response_model=Dict[str, Any])
async def get_filter_options(
    db: Session = Depends(get_db)
):
    """
    Get available filter options for job search.
    """
    try:
        # Get distinct values for filters
        industries = db.query(Job.industry).filter(
            Job.is_active == True,
            Job.industry.isnot(None)
        ).distinct().order_by(Job.industry).all()
        
        locations = db.query(Job.city).filter(
            Job.is_active == True,
            Job.city.isnot(None)
        ).distinct().order_by(Job.city).limit(50).all()
        
        companies = db.query(Job.company_name).filter(
            Job.is_active == True,
            Job.company_name.isnot(None)
        ).distinct().order_by(Job.company_name).limit(100).all()
        
        job_types = db.query(Job.job_type).filter(
            Job.is_active == True,
            Job.job_type.isnot(None)
        ).distinct().order_by(Job.job_type).all()
        
        experience_levels = db.query(Job.experience_level).filter(
            Job.is_active == True,
            Job.experience_level.isnot(None)
        ).distinct().order_by(Job.experience_level).all()
        
        return {
            "industries": [row[0] for row in industries if row[0]],
            "locations": [row[0] for row in locations if row[0]],
            "companies": [row[0] for row in companies if row[0]],
            "job_types": [row[0] for row in job_types if row[0]],
            "experience_levels": [row[0] for row in experience_levels if row[0]],
            "salary_ranges": [
                {"label": "0 - 5 Lakhs", "min": 0, "max": 500000},
                {"label": "5 - 10 Lakhs", "min": 500000, "max": 1000000},
                {"label": "10 - 15 Lakhs", "min": 1000000, "max": 1500000},
                {"label": "15 - 20 Lakhs", "min": 1500000, "max": 2000000},
                {"label": "20 - 30 Lakhs", "min": 2000000, "max": 3000000},
                {"label": "30+ Lakhs", "min": 3000000, "max": None}
            ],
            "boolean_filters": {
                "veteran_friendly": "Veteran-friendly jobs",
                "government_jobs": "Government jobs",
                "psu_jobs": "PSU jobs",
                "remote_only": "Remote work available",
                "security_clearance": "Security clearance required"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving filter options"
        )


# Helper functions

def create_user_profile_from_model(user: User):
    """Create UserProfile object from User model for job matching."""
    from app.ml.job_matcher import UserProfile
    
    return UserProfile(
        user_id=user.id,
        skills=(user.military_skills or []) + (user.civilian_skills or []) + (user.technical_skills or []),
        experience_years=user.total_experience_years or user.years_of_service or 0,
        preferred_locations=user.preferred_locations or [],
        current_location=user.current_location or '',
        preferred_industries=user.preferred_industries or [],
        preferred_job_titles=user.preferred_job_titles or [],
        expected_salary_min=user.expected_salary_min,
        expected_salary_max=user.expected_salary_max,
        willing_to_relocate=user.willing_to_relocate or False,
        preferred_work_type=user.preferred_work_type or 'full_time',
        service_branch=user.service_branch or '',
        rank=user.rank or '',
        military_skills=user.military_skills or [],
        civilian_skills=user.civilian_skills or [],
        technical_skills=user.technical_skills or []
    )


def get_similar_jobs(db: Session, job: Job, limit: int = 5) -> List[Job]:
    """Find similar jobs based on title, company, and industry."""
    similar_jobs = db.query(Job).filter(
        Job.is_active == True,
        Job.id != job.id,
        or_(
            Job.industry == job.industry,
            Job.company_name == job.company_name,
            Job.title.ilike(f"%{job.title[:20]}%")  # Similar title prefix
        )
    ).order_by(desc(Job.veteran_match_score)).limit(limit).all()
    
    return similar_jobs
