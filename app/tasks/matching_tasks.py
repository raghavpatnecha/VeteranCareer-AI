from celery import current_task
from celery.exceptions import Retry
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, func
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
from datetime import datetime, timedelta
import traceback
import json
import numpy as np
from collections import defaultdict

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.job import Job
from app.models.user import User
from app.models.application import Application
from app.ml.job_matcher import JobMatcher, UserProfile, JobMatchResult
from app.ml.career_advisor import CareerAdvisor
from app.ml.skill_translator import MilitarySkillTranslator
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def update_user_job_matches(self, user_id: int = None, batch_size: int = 50) -> Dict[str, Any]:
    """
    Update job matches for users based on their profiles and preferences.
    
    Args:
        user_id: Specific user ID to update (if None, updates all active users)
        batch_size: Number of users to process in each batch
        
    Returns:
        Dict with matching results and statistics
    """
    task_id = self.request.id
    logger.info(f"Starting job matching update task {task_id}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing job matcher'})
        
        # Initialize job matcher
        matcher = JobMatcher()
        db = SessionLocal()
        
        try:
            # Get users to process
            if user_id:
                users = db.query(User).filter(User.id == user_id, User.is_active == True).all()
            else:
                users = db.query(User).filter(User.is_active == True).all()
            
            total_users = len(users)
            processed_users = 0
            updated_matches = 0
            errors = 0
            
            logger.info(f"Processing job matches for {total_users} users")
            
            # Process users in batches
            for i in range(0, total_users, batch_size):
                batch_users = users[i:i + batch_size]
                
                for user in batch_users:
                    try:
                        # Update task progress
                        self.update_state(state='PROGRESS', meta={
                            'status': f'Processing user {user.id}',
                            'processed': processed_users,
                            'total': total_users
                        })
                        
                        # Calculate matches for this user
                        user_matches = calculate_user_job_matches.delay(user.id)
                        match_result = user_matches.get(timeout=300)  # 5 minutes timeout
                        
                        if match_result['status'] == 'completed':
                            updated_matches += match_result['matches_updated']
                        
                        processed_users += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing matches for user {user.id}: {str(e)}")
                        errors += 1
                        processed_users += 1
                        continue
                
                # Brief pause between batches
                if i + batch_size < total_users:
                    import time
                    time.sleep(2)
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'users_processed': processed_users,
                'matches_updated': updated_matches,
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Job matching update completed: {processed_users} users, {updated_matches} matches updated")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during matching update: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Job matching update task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying matching update task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def calculate_user_job_matches(self, user_id: int, job_limit: int = 100) -> Dict[str, Any]:
    """
    Calculate job matches for a specific user.
    
    Args:
        user_id: User ID to calculate matches for
        job_limit: Maximum number of jobs to evaluate
        
    Returns:
        Dict with matching results
    """
    task_id = self.request.id
    logger.info(f"Calculating job matches for user {user_id}, task {task_id}")
    
    try:
        # Initialize components
        matcher = JobMatcher()
        db = SessionLocal()
        
        try:
            # Get user profile
            user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
            if not user:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f'User {user_id} not found or inactive',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Create user profile for matching
            user_profile = create_user_profile(user)
            
            # Get active jobs for matching
            jobs_query = db.query(Job).filter(
                Job.is_active == True,
                Job.posted_date >= datetime.now() - timedelta(days=90)  # Recent jobs only
            ).order_by(desc(Job.posted_date))
            
            if job_limit:
                jobs = jobs_query.limit(job_limit).all()
            else:
                jobs = jobs_query.all()
            
            logger.info(f"Evaluating {len(jobs)} jobs for user {user_id}")
            
            # Calculate matches
            matches = []
            high_match_count = 0
            
            for job in jobs:
                try:
                    match_result = matcher.calculate_job_match_score(user_profile, job)
                    
                    # Store match if score is above threshold
                    if match_result.match_score >= 40:  # Minimum match threshold
                        matches.append({
                            'job_id': job.id,
                            'match_score': match_result.match_score,
                            'skill_match': match_result.skill_match_score,
                            'experience_match': match_result.experience_match_score,
                            'location_match': match_result.location_match_score,
                            'preference_match': match_result.preference_match_score,
                            'salary_match': match_result.salary_match_score,
                            'match_reasons': match_result.match_reasons,
                            'calculated_at': datetime.now().isoformat()
                        })
                        
                        if match_result.match_score >= 75:
                            high_match_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error calculating match for job {job.id}: {str(e)}")
                    continue
            
            # Sort matches by score
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Store top matches in user's profile or cache
            # For now, we'll just return the results
            # In production, you might want to store these in a separate table
            
            result = {
                'task_id': task_id,
                'user_id': user_id,
                'status': 'completed',
                'jobs_evaluated': len(jobs),
                'matches_found': len(matches),
                'high_matches': high_match_count,
                'matches_updated': len(matches),  # For compatibility with calling task
                'top_matches': matches[:20],  # Return top 20 matches
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Job matching completed for user {user_id}: {len(matches)} matches found")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during job matching: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Job matching calculation task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job matching task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'user_id': user_id,
            'status': 'failed', 
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=180)
def update_compatibility_scores(self, job_ids: List[int] = None, user_ids: List[int] = None) -> Dict[str, Any]:
    """
    Update compatibility scores for job-user combinations.
    
    Args:
        job_ids: Specific job IDs to update (if None, updates all active jobs)
        user_ids: Specific user IDs to update (if None, updates all active users)
        
    Returns:
        Dict with update results
    """
    task_id = self.request.id
    logger.info(f"Starting compatibility scores update task {task_id}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing compatibility calculator'})
        
        matcher = JobMatcher()
        db = SessionLocal()
        
        try:
            # Get jobs to process
            if job_ids:
                jobs = db.query(Job).filter(Job.id.in_(job_ids), Job.is_active == True).all()
            else:
                jobs = db.query(Job).filter(Job.is_active == True).limit(500).all()  # Limit for performance
            
            # Get users to process
            if user_ids:
                users = db.query(User).filter(User.id.in_(user_ids), User.is_active == True).all()
            else:
                users = db.query(User).filter(User.is_active == True).limit(100).all()  # Limit for performance
            
            total_combinations = len(jobs) * len(users)
            processed_combinations = 0
            high_compatibility_count = 0
            
            logger.info(f"Processing {total_combinations} job-user combinations")
            
            # Track compatibility scores
            compatibility_data = []
            
            for job in jobs:
                for user in users:
                    try:
                        # Create user profile
                        user_profile = create_user_profile(user)
                        
                        # Calculate compatibility
                        match_result = matcher.calculate_job_match_score(user_profile, job)
                        
                        # Store significant matches
                        if match_result.match_score >= 50:
                            compatibility_data.append({
                                'user_id': user.id,
                                'job_id': job.id,
                                'compatibility_score': match_result.match_score,
                                'component_scores': {
                                    'skills': match_result.skill_match_score,
                                    'experience': match_result.experience_match_score,
                                    'location': match_result.location_match_score,
                                    'preferences': match_result.preference_match_score,
                                    'salary': match_result.salary_match_score
                                },
                                'calculated_at': datetime.now().isoformat()
                            })
                            
                            if match_result.match_score >= 80:
                                high_compatibility_count += 1
                        
                        processed_combinations += 1
                        
                        # Update progress every 100 combinations
                        if processed_combinations % 100 == 0:
                            self.update_state(state='PROGRESS', meta={
                                'status': 'Calculating compatibility scores',
                                'processed': processed_combinations,
                                'total': total_combinations
                            })
                        
                    except Exception as e:
                        logger.warning(f"Error calculating compatibility for user {user.id} and job {job.id}: {str(e)}")
                        processed_combinations += 1
                        continue
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'jobs_processed': len(jobs),
                'users_processed': len(users),
                'total_combinations': total_combinations,
                'significant_matches': len(compatibility_data),
                'high_compatibility_matches': high_compatibility_count,
                'compatibility_data': compatibility_data[:100],  # Return top 100 for response
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Compatibility scores update completed: {len(compatibility_data)} significant matches")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during compatibility update: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Compatibility scores update task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying compatibility update task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=240)
def generate_user_recommendations(self, user_id: int, recommendation_types: List[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive recommendations for a user.
    
    Args:
        user_id: User ID to generate recommendations for
        recommendation_types: Types of recommendations to generate
        
    Returns:
        Dict with recommendation results
    """
    task_id = self.request.id
    logger.info(f"Generating recommendations for user {user_id}, task {task_id}")
    
    try:
        # Default recommendation types
        if not recommendation_types:
            recommendation_types = ['jobs', 'skills', 'career_advice', 'industry_trends']
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing recommendation engines'})
        
        # Initialize components
        matcher = JobMatcher()
        career_advisor = CareerAdvisor()
        skill_translator = MilitarySkillTranslator()
        
        db = SessionLocal()
        
        try:
            # Get user profile
            user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
            if not user:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f'User {user_id} not found or inactive',
                    'timestamp': datetime.now().isoformat()
                }
            
            recommendations = {}
            
            # Job recommendations
            if 'jobs' in recommendation_types:
                self.update_state(state='PROGRESS', meta={'status': 'Generating job recommendations'})
                
                job_recs = calculate_user_job_matches.delay(user_id, job_limit=50)
                job_results = job_recs.get(timeout=300)
                
                recommendations['jobs'] = {
                    'top_matches': job_results.get('top_matches', [])[:10],
                    'total_matches': job_results.get('matches_found', 0),
                    'high_matches': job_results.get('high_matches', 0)
                }
            
            # Skill recommendations
            if 'skills' in recommendation_types:
                self.update_state(state='PROGRESS', meta={'status': 'Analyzing skill gaps'})
                
                skill_recs = analyze_skill_gaps.delay(user_id)
                skill_results = skill_recs.get(timeout=180)
                
                recommendations['skills'] = skill_results.get('recommendations', {})
            
            # Career advice
            if 'career_advice' in recommendation_types:
                self.update_state(state='PROGRESS', meta={'status': 'Generating career advice'})
                
                user_profile_dict = {
                    'user_id': user.id,
                    'military_skills': user.military_skills or [],
                    'civilian_skills': user.civilian_skills or [],
                    'technical_skills': user.technical_skills or [], 
                    'years_of_service': user.years_of_service or 0,
                    'service_branch': user.service_branch or '',
                    'rank': user.rank or '',
                    'current_location': user.current_location or '',
                    'preferred_industries': user.preferred_industries or [],
                    'expected_salary_min': user.expected_salary_min,
                    'expected_salary_max': user.expected_salary_max
                }
                
                career_advice = career_advisor.generate_career_advice(user_profile_dict)
                
                recommendations['career_advice'] = {
                    'recommended_industries': career_advice.recommended_industries[:5],
                    'career_paths': career_advice.career_paths[:3],
                    'skill_gaps': career_advice.skill_gaps[:10],
                    'next_steps': career_advice.next_steps[:5],
                    'salary_insights': career_advice.salary_insights
                }
            
            # Industry trends
            if 'industry_trends' in recommendation_types:
                self.update_state(state='PROGRESS', meta={'status': 'Analyzing industry trends'})
                
                # Get trending skills and industries
                trending_skills = get_trending_skills_for_veterans()
                industry_analysis = analyze_veteran_industry_trends()
                
                recommendations['industry_trends'] = {
                    'trending_skills': trending_skills[:10],
                    'growing_industries': industry_analysis.get('growing_industries', [])[:5],
                    'market_insights': industry_analysis.get('market_insights', {})
                }
            
            result = {
                'task_id': task_id,
                'user_id': user_id,
                'status': 'completed',
                'recommendations': recommendations,
                'recommendation_types': recommendation_types,
                'generated_at': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Recommendations generated for user {user_id}")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during recommendation generation: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Recommendation generation task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying recommendation task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=90 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'user_id': user_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def analyze_skill_gaps(self, user_id: int) -> Dict[str, Any]:
    """
    Analyze skill gaps for a user based on target jobs and industry requirements.
    
    Args:
        user_id: User ID to analyze skill gaps for
        
    Returns:
        Dict with skill gap analysis
    """
    task_id = self.request.id
    logger.info(f"Analyzing skill gaps for user {user_id}, task {task_id}")
    
    try:
        skill_translator = MilitarySkillTranslator()
        db = SessionLocal()
        
        try:
            # Get user profile
            user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
            if not user:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f'User {user_id} not found or inactive',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get user's current skills
            current_skills = []
            current_skills.extend(user.military_skills or [])
            current_skills.extend(user.civilian_skills or [])
            current_skills.extend(user.technical_skills or [])
            
            # Get jobs user is interested in (high-match jobs)
            target_jobs = db.query(Job).filter(
                Job.is_active == True,
                Job.veteran_match_score >= 60
            ).limit(20).all()
            
            # Analyze required skills across target jobs
            required_skills = defaultdict(int)
            for job in target_jobs:
                for skill in (job.required_skills or []):
                    required_skills[skill.lower().strip()] += 1
            
            # Identify gaps
            current_skills_lower = set(skill.lower().strip() for skill in current_skills)
            skill_gaps = []
            
            for skill, frequency in required_skills.items():
                if skill not in current_skills_lower and frequency >= 3:  # Required by at least 3 jobs
                    skill_gaps.append({
                        'skill': skill,
                        'frequency': frequency,
                        'importance': 'high' if frequency >= 10 else 'medium' if frequency >= 6 else 'low',
                        'category': categorize_skill(skill)
                    })
            
            # Sort by frequency (importance)
            skill_gaps.sort(key=lambda x: x['frequency'], reverse=True)
            
            # Get skill translations and recommendations
            recommendations = []
            for gap in skill_gaps[:10]:  # Top 10 gaps
                translation = skill_translator.translate_skill(gap['skill'])
                recommendations.append({
                    'skill': gap['skill'],
                    'frequency': gap['frequency'],
                    'importance': gap['importance'],
                    'category': gap['category'],
                    'related_industries': translation.applicable_industries[:3],
                    'related_job_titles': translation.related_job_titles[:3],
                    'transferability_notes': translation.transferability_notes
                })
            
            result = {
                'task_id': task_id,
                'user_id': user_id,
                'status': 'completed',
                'current_skills_count': len(current_skills),
                'target_jobs_analyzed': len(target_jobs),
                'skill_gaps_identified': len(skill_gaps),
                'recommendations': {
                    'priority_skills': recommendations[:5],
                    'secondary_skills': recommendations[5:10],
                    'skill_categories': get_skill_category_analysis(skill_gaps),
                    'learning_suggestions': generate_learning_suggestions(recommendations[:5])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Skill gap analysis completed for user {user_id}: {len(skill_gaps)} gaps identified")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during skill gap analysis: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Skill gap analysis task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying skill gap analysis task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'user_id': user_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


# Helper functions

def create_user_profile(user: User) -> UserProfile:
    """Create UserProfile object from User model."""
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


def categorize_skill(skill: str) -> str:
    """Categorize skill into broad categories."""
    skill_lower = skill.lower()
    
    technical_keywords = ['programming', 'software', 'database', 'cloud', 'network', 'security', 'data', 'analytics']
    leadership_keywords = ['management', 'leadership', 'team', 'project', 'supervision', 'coordination']
    business_keywords = ['marketing', 'sales', 'finance', 'strategy', 'business', 'operations']
    
    if any(keyword in skill_lower for keyword in technical_keywords):
        return 'technical'
    elif any(keyword in skill_lower for keyword in leadership_keywords):
        return 'leadership'  
    elif any(keyword in skill_lower for keyword in business_keywords):
        return 'business'
    else:
        return 'domain_specific'


def get_skill_category_analysis(skill_gaps: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze skill gaps by category."""
    categories = defaultdict(int)
    for gap in skill_gaps:
        categories[gap['category']] += 1
    return dict(categories)


def generate_learning_suggestions(priority_skills: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate learning suggestions for priority skills."""
    suggestions = []
    
    for skill_info in priority_skills:
        skill = skill_info['skill']
        category = skill_info['category']
        
        if category == 'technical':
            suggestions.append({
                'skill': skill,
                'suggestion': f"Consider online courses or certifications in {skill}",
                'resources': "Coursera, Udemy, or industry-specific training programs"
            })
        elif category == 'leadership':
            suggestions.append({
                'skill': skill,
                'suggestion': f"Leverage military leadership experience to demonstrate {skill}",
                'resources': "Management training programs, executive coaching"
            })
        else:
            suggestions.append({
                'skill': skill,
                'suggestion': f"Gain practical experience in {skill} through projects or internships",
                'resources': "Industry workshops, professional associations"
            })
    
    return suggestions


def get_trending_skills_for_veterans() -> List[Dict[str, Any]]:
    """Get trending skills specifically relevant to veterans."""
    # This would typically query job market data
    # For now, returning static data based on current market trends
    return [
        {'skill': 'cybersecurity', 'growth_rate': 0.25, 'veteran_suitability': 0.95},
        {'skill': 'project management', 'growth_rate': 0.15, 'veteran_suitability': 0.90},
        {'skill': 'cloud computing', 'growth_rate': 0.30, 'veteran_suitability': 0.80},
        {'skill': 'data analysis', 'growth_rate': 0.20, 'veteran_suitability': 0.85},
        {'skill': 'logistics optimization', 'growth_rate': 0.18, 'veteran_suitability': 0.95},
        {'skill': 'quality assurance', 'growth_rate': 0.12, 'veteran_suitability': 0.88},
        {'skill': 'compliance management', 'growth_rate': 0.14, 'veteran_suitability': 0.92},
        {'skill': 'operations management', 'growth_rate': 0.10, 'veteran_suitability': 0.90},
        {'skill': 'risk assessment', 'growth_rate': 0.16, 'veteran_suitability': 0.93},
        {'skill': 'team leadership', 'growth_rate': 0.08, 'veteran_suitability': 0.95}
    ]


def analyze_veteran_industry_trends() -> Dict[str, Any]:
    """Analyze industry trends relevant to veterans."""
    return {
        'growing_industries': [
            {'industry': 'cybersecurity', 'growth_rate': 0.22, 'veteran_demand': 'very_high'},
            {'industry': 'defense_contracting', 'growth_rate': 0.08, 'veteran_demand': 'very_high'},
            {'industry': 'logistics', 'growth_rate': 0.12, 'veteran_demand': 'high'},
            {'industry': 'project_management', 'growth_rate': 0.10, 'veteran_demand': 'high'},
            {'industry': 'government_services', 'growth_rate': 0.05, 'veteran_demand': 'very_high'}
        ],
        'market_insights': {
            'remote_work_trend': 0.45,
            'skill_shortage_areas': ['cybersecurity', 'project management', 'data analysis'],
            'salary_growth_sectors': ['technology', 'cybersecurity', 'healthcare'],
            'veteran_hiring_initiatives': 'increasing'
        }
    }


# Task monitoring and utility functions

@celery_app.task
def get_matching_task_status() -> Dict[str, Any]:
    """Get status of all matching-related tasks."""
    db = None
    try:
        # Get active tasks from Celery
        active_tasks = celery_app.control.inspect().active()
        scheduled_tasks = celery_app.control.inspect().scheduled()
        
        matching_tasks = []
        task_types = ['update_user_job_matches', 'calculate_user_job_matches', 'update_compatibility_scores', 
                     'generate_user_recommendations', 'analyze_skill_gaps']
        
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    if any(task_type in task['name'] for task_type in task_types):
                        matching_tasks.append({
                            'worker': worker,
                            'task_id': task['id'],
                            'name': task['name'],
                            'status': 'active',
                            'args': task.get('args', []),
                            'kwargs': task.get('kwargs', {})
                        })
        
        # Get database statistics
        db = SessionLocal()
        
        # Count recent matching activities
        recent_applications = db.query(Application).filter(
            Application.created_at >= datetime.now() - timedelta(hours=24)
        ).count()
        
        active_users = db.query(User).filter(User.is_active == True).count()
        active_jobs = db.query(Job).filter(Job.is_active == True).count()
        
        return {
            'status': 'success',
            'active_matching_tasks': len(matching_tasks),
            'tasks': matching_tasks,
            'statistics': {
                'active_users': active_users,
                'active_jobs': active_jobs,
                'recent_applications': recent_applications
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting matching task status: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    finally:
        if db:
            db.close()