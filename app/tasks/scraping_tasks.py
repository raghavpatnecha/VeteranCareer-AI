from celery import current_task
from celery.exceptions import Retry
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime, timedelta
import traceback
import json

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.job import Job
from app.models.user import User
from app.scrapers.scraper_manager import get_scraper_manager
from app.scrapers.job_portal_scraper import JobPortalScraper
from app.scrapers.psu_scraper import PSUScraper
from app.ml.data_processor import JobDataProcessor, ProcessingConfig
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def run_job_scraping(self, portals: List[str] = None, max_jobs: int = 200) -> Dict[str, Any]:
    """
    Celery task to run job portal scraping.
    
    Args:
        portals: List of job portals to scrape (default: all)
        max_jobs: Maximum number of jobs to scrape per portal
        
    Returns:
        Dict with scraping results and statistics
    """
    task_id = self.request.id
    logger.info(f"Starting job scraping task {task_id}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing job portal scraper'})
        
        # Initialize scraper
        scraper = JobPortalScraper()
        
        # Set default portals if none provided
        if portals is None:
            portals = ["naukri", "indeed", "monster", "foundit"]
        
        # Configure scraping parameters
        scraping_params = {
            "keywords": "ex-servicemen OR military OR veteran OR government OR defence",
            "location": "india",
            "portals": portals,
            "max_pages": max(1, max_jobs // 20)  # Approximate pages based on max_jobs
        }
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Starting scraping process'})
        
        # Run scraping (need to handle async in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize scraper
            loop.run_until_complete(scraper.start())
            
            # Run scraping
            scraped_jobs = loop.run_until_complete(scraper.scrape_jobs(**scraping_params))
            
            # Stop scraper
            loop.run_until_complete(scraper.stop())
            
        finally:
            loop.close()
        
        logger.info(f"Scraped {len(scraped_jobs)} jobs from portals: {portals}")
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Processing scraped data'})
        
        # Process and save scraped data
        result = process_and_save_jobs.delay(scraped_jobs, source_task=task_id)
        processing_result = result.get(timeout=600)  # 10 minutes timeout
        
        # Combine results
        final_result = {
            'task_id': task_id,
            'status': 'completed',
            'portals_scraped': portals,
            'jobs_scraped': len(scraped_jobs),
            'processing_result': processing_result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Job scraping task {task_id} completed successfully")
        return final_result
        
    except Exception as exc:
        logger.error(f"Job scraping task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job scraping task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def run_psu_scraping(self, organizations: List[str] = None, max_jobs_per_org: int = 20) -> Dict[str, Any]:
    """
    Celery task to run PSU/Government organization scraping.
    
    Args:
        organizations: List of organizations to scrape (default: all)
        max_jobs_per_org: Maximum jobs to scrape per organization
        
    Returns:
        Dict with scraping results and statistics
    """
    task_id = self.request.id
    logger.info(f"Starting PSU scraping task {task_id}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing PSU scraper'})
        
        # Initialize scraper
        scraper = PSUScraper()
        
        # Set default organizations if none provided
        if organizations is None:
            organizations = ["hal", "bel", "drdo", "isro", "ongc", "ntpc", "bhel", "gail"]
        
        # Configure scraping parameters
        scraping_params = {
            "organizations": organizations,
            "include_ex_servicemen_only": True,
            "max_jobs_per_org": max_jobs_per_org
        }
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Starting PSU scraping process'})
        
        # Run scraping (need to handle async in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize scraper
            loop.run_until_complete(scraper.start())
            
            # Run scraping
            scraped_jobs = loop.run_until_complete(scraper.scrape_jobs(**scraping_params))
            
            # Stop scraper
            loop.run_until_complete(scraper.stop())
            
        finally:
            loop.close()
        
        logger.info(f"Scraped {len(scraped_jobs)} jobs from PSU organizations: {organizations}")
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Processing scraped PSU data'})
        
        # Process and save scraped data
        result = process_and_save_jobs.delay(scraped_jobs, source_task=task_id)
        processing_result = result.get(timeout=600)  # 10 minutes timeout
        
        # Combine results
        final_result = {
            'task_id': task_id,
            'status': 'completed',
            'organizations_scraped': organizations,
            'jobs_scraped': len(scraped_jobs),
            'processing_result': processing_result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"PSU scraping task {task_id} completed successfully")
        return final_result
        
    except Exception as exc:
        logger.error(f"PSU scraping task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying PSU scraping task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def process_and_save_jobs(self, jobs_data: List[Dict[str, Any]], source_task: str = None) -> Dict[str, Any]:
    """
    Process scraped job data and save to database.
    
    Args:
        jobs_data: List of scraped job dictionaries
        source_task: ID of the source scraping task
        
    Returns:
        Dict with processing results and statistics
    """
    task_id = self.request.id
    logger.info(f"Starting job processing task {task_id} for {len(jobs_data)} jobs")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'status': 'Processing job data',
            'total_jobs': len(jobs_data),
            'processed': 0
        })
        
        # Initialize data processor
        processor = JobDataProcessor()
        
        # Convert to DataFrame for processing
        import pandas as pd
        df = pd.DataFrame(jobs_data)
        
        # Process the data
        processing_result = processor.process_job_data(df)
        processed_df = processing_result.processed_data
        
        logger.info(f"Processed {len(processed_df)} jobs (from {len(jobs_data)} original)")
        
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'status': 'Saving to database',
            'total_jobs': len(processed_df),
            'processed': 0
        })
        
        # Save to database
        db = SessionLocal()
        saved_count = 0
        updated_count = 0
        duplicate_count = 0
        error_count = 0
        
        try:
            for idx, (_, row) in enumerate(processed_df.iterrows()):
                try:
                    # Check if job already exists
                    existing_job = db.query(Job).filter(
                        Job.title == row.get('title', ''),
                        Job.company_name == row.get('company', ''),
                        Job.location == row.get('location', '')
                    ).first()
                    
                    if existing_job:
                        # Update existing job with new information
                        if update_existing_job(existing_job, row):
                            updated_count += 1
                        else:
                            duplicate_count += 1
                    else:
                        # Create new job
                        if create_new_job(db, row):
                            saved_count += 1
                    
                    # Update progress every 10 jobs
                    if (idx + 1) % 10 == 0:
                        self.update_state(state='PROGRESS', meta={
                            'status': 'Saving to database',
                            'total_jobs': len(processed_df),
                            'processed': idx + 1
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing job {idx}: {str(e)}")
                    error_count += 1
                    continue
            
            # Commit all changes
            db.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
        
        # Final results
        result = {
            'task_id': task_id,
            'source_task': source_task,
            'status': 'completed',
            'total_jobs_processed': len(jobs_data),
            'jobs_after_processing': len(processed_df),
            'jobs_saved': saved_count,
            'jobs_updated': updated_count,
            'jobs_duplicated': duplicate_count,
            'processing_errors': error_count,
            'processing_quality_metrics': processing_result.quality_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Job processing task {task_id} completed: {saved_count} saved, {updated_count} updated")
        return result
        
    except Exception as exc:
        logger.error(f"Job processing task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job processing task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'source_task': source_task,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def cleanup_expired_jobs(self, days_old: int = 30) -> Dict[str, Any]:
    """
    Clean up expired and old job postings.
    
    Args:
        days_old: Jobs older than this many days will be marked inactive
        
    Returns:
        Dict with cleanup results
    """
    task_id = self.request.id
    logger.info(f"Starting job cleanup task {task_id}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Starting cleanup process'})
        
        db = SessionLocal()
        cleanup_date = datetime.now() - timedelta(days=days_old)
        
        try:
            # Mark old jobs as inactive
            old_jobs = db.query(Job).filter(
                Job.created_at < cleanup_date,
                Job.is_active == True
            ).all()
            
            inactive_count = 0
            for job in old_jobs:
                job.is_active = False
                inactive_count += 1
            
            # Delete very old inactive jobs (90+ days)
            very_old_date = datetime.now() - timedelta(days=90)
            deleted_jobs = db.query(Job).filter(
                Job.created_at < very_old_date,
                Job.is_active == False
            )
            deleted_count = deleted_jobs.count()
            deleted_jobs.delete()
            
            # Mark jobs with expired deadlines as inactive
            expired_jobs = db.query(Job).filter(
                Job.application_deadline < datetime.now(),
                Job.is_active == True
            ).all()
            
            expired_count = 0
            for job in expired_jobs:
                job.is_active = False
                expired_count += 1
            
            db.commit()
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'jobs_marked_inactive': inactive_count,
                'jobs_deleted': deleted_count,
                'expired_jobs_deactivated': expired_count,
                'cleanup_date_threshold': cleanup_date.isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Job cleanup completed: {inactive_count} deactivated, {deleted_count} deleted")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during cleanup: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Job cleanup task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying cleanup task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=180)
def update_job_matching_scores(self, batch_size: int = 100) -> Dict[str, Any]:
    """
    Update job matching scores for all active jobs.
    
    Args:
        batch_size: Number of jobs to process in each batch
        
    Returns:
        Dict with update results
    """
    task_id = self.request.id
    logger.info(f"Starting job matching scores update task {task_id}")
    
    try:
        from app.ml.job_matcher import JobMatcher
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing job matcher'})
        
        matcher = JobMatcher()
        db = SessionLocal()
        
        try:
            # Get all active jobs that need score updates
            jobs_query = db.query(Job).filter(Job.is_active == True)
            total_jobs = jobs_query.count()
            
            updated_count = 0
            error_count = 0
            
            # Process jobs in batches
            for offset in range(0, total_jobs, batch_size):
                batch_jobs = jobs_query.offset(offset).limit(batch_size).all()
                
                for job in batch_jobs:
                    try:
                        # Calculate new veteran match score
                        # This is a simplified version - in practice you'd need user profiles
                        new_score = calculate_basic_veteran_score(job)
                        
                        if abs(job.veteran_match_score - new_score) > 5:  # Only update if significant change
                            job.veteran_match_score = new_score
                            job.last_updated = datetime.now()
                            updated_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error updating job {job.id}: {str(e)}")
                        error_count += 1
                        continue
                
                # Update progress
                self.update_state(state='PROGRESS', meta={
                    'status': 'Updating job scores',
                    'processed': min(offset + batch_size, total_jobs),
                    'total': total_jobs
                })
                
                # Commit batch
                db.commit()
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'total_jobs_processed': total_jobs,
                'jobs_updated': updated_count,
                'errors': error_count,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Job matching scores update completed: {updated_count} jobs updated")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during score update: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Job matching scores update task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying score update task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=90 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


def create_new_job(db: Session, job_data: Dict[str, Any]) -> bool:
    """Create a new job record from processed data."""
    try:
        job = Job(
            title=job_data.get('title', ''),
            company_name=job_data.get('company', job_data.get('company_name', '')),
            location=job_data.get('location_normalized', job_data.get('location', '')),
            description=job_data.get('description', ''),
            requirements=job_data.get('requirements', ''),
            job_type=job_data.get('job_type', 'full_time'),
            source_portal=job_data.get('source_portal', job_data.get('source_organization', '')),
            source_url=job_data.get('job_url', ''),
            external_job_id=job_data.get('external_id', ''),
            
            # Parse dates
            posted_date=parse_date(job_data.get('posted_date')),
            application_deadline=parse_date(job_data.get('application_deadline')),
            
            # Veteran-specific fields
            veteran_preference=job_data.get('is_veteran_friendly', job_data.get('is_ex_servicemen_eligible', False)),
            veteran_match_score=job_data.get('veteran_match_score', 0.0),
            government_job=job_data.get('is_government_job', False),
            psu_job=job_data.get('is_psu_job', False),
            
            # Skills and requirements
            required_skills=job_data.get('skills', []),
            education_requirements=job_data.get('qualification', '').split(',') if job_data.get('qualification') else [],
            
            # Salary information
            salary_min=job_data.get('salary_min'),
            salary_max=job_data.get('salary_max'),
            
            # Experience requirements
            min_experience_years=extract_experience_years(job_data.get('experience', ''), 'min'),
            max_experience_years=extract_experience_years(job_data.get('experience', ''), 'max'),
            
            # Default values
            is_active=True,
            view_count=0,
            scraped_at=datetime.now()
        )
        
        db.add(job)
        return True
        
    except Exception as e:
        logger.error(f"Error creating new job: {str(e)}")
        return False


def update_existing_job(job: Job, job_data: Dict[str, Any]) -> bool:
    """Update existing job with new information."""
    try:
        updated = False
        
        # Update description if empty
        if not job.description and job_data.get('description'):
            job.description = job_data['description']
            updated = True
        
        # Update requirements if empty
        if not job.requirements and job_data.get('requirements'):
            job.requirements = job_data['requirements']
            updated = True
        
        # Update skills if empty
        if not job.required_skills and job_data.get('skills'):
            job.required_skills = job_data['skills']
            updated = True
        
        # Update veteran matching score if higher
        new_score = job_data.get('veteran_match_score', 0.0)
        if new_score > job.veteran_match_score:
            job.veteran_match_score = new_score
            updated = True
        
        # Update last seen timestamp
        job.last_updated = datetime.now()
        
        return updated
        
    except Exception as e:
        logger.error(f"Error updating existing job: {str(e)}")
        return False


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object."""
    if not date_str:
        return None
    
    try:
        # Try different date formats
        from dateutil import parser
        return parser.parse(date_str)
    except:
        return None


def extract_experience_years(experience_str: str, part: str) -> Optional[int]:
    """Extract experience years from string."""
    if not experience_str:
        return None
    
    try:
        import re
        # Extract numbers from experience string
        numbers = re.findall(r'\d+', experience_str)
        
        if not numbers:
            return None
        
        if len(numbers) == 1:
            return int(numbers[0]) if part == 'min' else int(numbers[0])
        else:
            return int(numbers[0]) if part == 'min' else int(numbers[1])
            
    except:
        return None


def calculate_basic_veteran_score(job: Job) -> float:
    """Calculate basic veteran match score for a job."""
    score = 40.0  # Base score
    
    # Veteran preference jobs get high score
    if job.veteran_preference:
        score += 30.0
    
    # Government/PSU jobs are good for veterans
    if job.government_job or job.psu_job:
        score += 20.0
    
    # Check job title for veteran-friendly roles
    title_lower = job.title.lower() if job.title else ''
    veteran_friendly_titles = [
        'manager', 'officer', 'supervisor', 'coordinator', 'analyst',
        'security', 'operations', 'logistics', 'maintenance',
        'project', 'team lead', 'administrator'
    ]
    
    if any(title in title_lower for title in veteran_friendly_titles):
        score += 10.0
    
    # Security clearance advantage
    if job.description and 'security clearance' in job.description.lower():
        score += 15.0
    
    return min(100.0, max(0.0, score))


# Task monitoring and status functions
@celery_app.task
def get_scraping_status() -> Dict[str, Any]:
    """Get current status of all scraping tasks."""
    try:
        # Get active tasks from Celery
        active_tasks = celery_app.control.inspect().active()
        scheduled_tasks = celery_app.control.inspect().scheduled()
        
        scraping_tasks = []
        task_types = ['run_job_scraping', 'run_psu_scraping', 'process_and_save_jobs']
        
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    if any(task_type in task['name'] for task_type in task_types):
                        scraping_tasks.append({
                            'worker': worker,
                            'task_id': task['id'],
                            'name': task['name'],
                            'status': 'active',
                            'args': task.get('args', []),
                            'kwargs': task.get('kwargs', {})
                        })
        
        return {
            'status': 'success',
            'active_scraping_tasks': len(scraping_tasks),
            'tasks': scraping_tasks,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task
def trigger_manual_scraping(portals: List[str] = None, organizations: List[str] = None) -> Dict[str, Any]:
    """Trigger manual scraping of specified portals and organizations."""
    try:
        task_ids = []
        
        # Trigger job portal scraping if requested
        if portals:
            job_task = run_job_scraping.delay(portals=portals)
            task_ids.append({
                'type': 'job_portal_scraping',
                'task_id': job_task.id,
                'portals': portals
            })
        
        # Trigger PSU scraping if requested
        if organizations:
            psu_task = run_psu_scraping.delay(organizations=organizations)
            task_ids.append({
                'type': 'psu_scraping',
                'task_id': psu_task.id,
                'organizations': organizations
            })
        
        return {
            'status': 'success',
            'message': 'Manual scraping tasks triggered',
            'tasks': task_ids,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering manual scraping: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
