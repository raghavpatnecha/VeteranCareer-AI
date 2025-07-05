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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from jinja2 import Template
from collections import defaultdict

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.job import Job
from app.models.user import User
from app.models.application import Application, ApplicationStatus
from app.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def send_daily_job_alerts(self, force_send: bool = False) -> Dict[str, Any]:
    """
    Send daily job alerts to users based on their preferences.
    
    Args:
        force_send: Send alerts even if not the scheduled time
        
    Returns:
        Dict with sending results and statistics
    """
    task_id = self.request.id
    logger.info(f"Starting daily job alerts task {task_id}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Initializing job alert system'})
        
        db = SessionLocal()
        
        try:
            # Get users who want daily job alerts
            users = db.query(User).filter(
                User.is_active == True,
                User.email_notifications == True,
                User.job_alert_frequency == 'daily'
            ).all()
            
            logger.info(f"Processing daily job alerts for {len(users)} users")
            
            sent_count = 0
            error_count = 0
            total_jobs_sent = 0
            
            # Get new jobs from the last 24 hours
            yesterday = datetime.now() - timedelta(days=1)
            new_jobs = db.query(Job).filter(
                Job.is_active == True,
                Job.created_at >= yesterday,
                Job.veteran_match_score >= 50  # Only send relevant jobs
            ).all()
            
            if not new_jobs and not force_send:
                logger.info("No new jobs found for daily alerts")
                return {
                    'task_id': task_id,
                    'status': 'completed',
                    'message': 'No new jobs to send',
                    'users_processed': 0,
                    'alerts_sent': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update task state
            self.update_state(state='PROGRESS', meta={
                'status': 'Sending job alerts',
                'users_to_process': len(users),
                'new_jobs_found': len(new_jobs)
            })
            
            for idx, user in enumerate(users):
                try:
                    # Update progress
                    if idx % 10 == 0:
                        self.update_state(state='PROGRESS', meta={
                            'status': f'Processing user {idx + 1} of {len(users)}',
                            'sent_count': sent_count,
                            'error_count': error_count
                        })
                    
                    # Filter jobs based on user preferences
                    relevant_jobs = filter_jobs_for_user(user, new_jobs)
                    
                    if relevant_jobs:
                        # Send job alert email
                        alert_result = send_job_alert_email.delay(
                            user.id, 
                            [job.id for job in relevant_jobs],
                            'daily'
                        )
                        
                        # Wait for result with timeout
                        try:
                            result = alert_result.get(timeout=60)
                            if result['status'] == 'sent':
                                sent_count += 1
                                total_jobs_sent += len(relevant_jobs)
                            else:
                                error_count += 1
                        except Exception as e:
                            logger.error(f"Error getting alert result for user {user.id}: {str(e)}")
                            error_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing job alert for user {user.id}: {str(e)}")
                    error_count += 1
                    continue
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'users_processed': len(users),
                'alerts_sent': sent_count,
                'alerts_failed': error_count,
                'total_jobs_sent': total_jobs_sent,
                'new_jobs_found': len(new_jobs),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Daily job alerts completed: {sent_count} sent, {error_count} failed")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during job alerts: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Daily job alerts task {task_id} failed: {str(exc)}")
        logger.error(traceback.format_exc())
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job alerts task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def send_job_alert_email(self, user_id: int, job_ids: List[int], alert_type: str = 'daily') -> Dict[str, Any]:
    """
    Send job alert email to a specific user.
    
    Args:
        user_id: User ID to send alert to
        job_ids: List of job IDs to include in alert
        alert_type: Type of alert (daily, weekly, immediate)
        
    Returns:
        Dict with sending result
    """
    task_id = self.request.id
    logger.info(f"Sending job alert email to user {user_id}, task {task_id}")
    
    try:
        db = SessionLocal()
        
        try:
            # Get user
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f'User {user_id} not found',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get jobs
            jobs = db.query(Job).filter(Job.id.in_(job_ids)).all()
            if not jobs:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': 'No jobs found for alert',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate email content
            email_content = generate_job_alert_email_content(user, jobs, alert_type)
            
            # Send email
            email_result = send_email(
                to_email=user.email,
                subject=email_content['subject'],
                html_content=email_content['html'],
                text_content=email_content['text']
            )
            
            if email_result['success']:
                result = {
                    'task_id': task_id,
                    'user_id': user_id,
                    'status': 'sent',
                    'jobs_count': len(jobs),
                    'alert_type': alert_type,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'task_id': task_id,
                    'user_id': user_id,
                    'status': 'failed',
                    'error': email_result.get('error', 'Email sending failed'),
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"Job alert email result for user {user_id}: {result['status']}")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during email sending: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Job alert email task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying email task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'user_id': user_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=180)
def send_application_reminders(self, reminder_type: str = 'follow_up') -> Dict[str, Any]:
    """
    Send application follow-up and status reminders to users.
    
    Args:
        reminder_type: Type of reminder (follow_up, interview_prep, status_check)
        
    Returns:
        Dict with reminder results
    """
    task_id = self.request.id
    logger.info(f"Starting application reminders task {task_id}, type: {reminder_type}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Finding applications needing reminders'})
        
        db = SessionLocal()
        
        try:
            reminders_sent = 0
            errors = 0
            
            if reminder_type == 'follow_up':
                # Find applications that need follow-up
                cutoff_date = datetime.now() - timedelta(days=7)
                applications = db.query(Application).join(User).filter(
                    Application.status.in_([ApplicationStatus.APPLIED, ApplicationStatus.VIEWED]),
                    Application.applied_date <= cutoff_date,
                    User.email_notifications == True,
                    User.is_active == True
                ).all()
                
            elif reminder_type == 'interview_prep':
                # Find applications with interviews in next 24-48 hours
                tomorrow = datetime.now() + timedelta(days=1)
                day_after = datetime.now() + timedelta(days=2)
                applications = db.query(Application).join(User).filter(
                    Application.status == ApplicationStatus.INTERVIEW_SCHEDULED,
                    Application.interview_scheduled_date >= tomorrow,
                    Application.interview_scheduled_date <= day_after,
                    User.email_notifications == True,
                    User.is_active == True
                ).all()
                
            elif reminder_type == 'status_check':
                # Find applications pending response for 2+ weeks
                cutoff_date = datetime.now() - timedelta(days=14)
                applications = db.query(Application).join(User).filter(
                    Application.status.in_([ApplicationStatus.INTERVIEW_COMPLETED, ApplicationStatus.UNDER_REVIEW]),
                    Application.status_updated_at <= cutoff_date,
                    User.email_notifications == True,
                    User.is_active == True
                ).all()
            else:
                applications = []
            
            logger.info(f"Found {len(applications)} applications for {reminder_type} reminders")
            
            # Update task state
            self.update_state(state='PROGRESS', meta={
                'status': 'Sending reminders',
                'applications_found': len(applications)
            })
            
            for idx, application in enumerate(applications):
                try:
                    # Update progress
                    if idx % 10 == 0:
                        self.update_state(state='PROGRESS', meta={
                            'status': f'Processing application {idx + 1} of {len(applications)}',
                            'sent_count': reminders_sent,
                            'error_count': errors
                        })
                    
                    # Send reminder
                    reminder_result = send_application_reminder_email.delay(
                        application.id,
                        reminder_type
                    )
                    
                    # Wait for result
                    try:
                        result = reminder_result.get(timeout=60)
                        if result['status'] == 'sent':
                            reminders_sent += 1
                        else:
                            errors += 1
                    except Exception as e:
                        logger.error(f"Error getting reminder result for application {application.id}: {str(e)}")
                        errors += 1
                
                except Exception as e:
                    logger.error(f"Error processing reminder for application {application.id}: {str(e)}")
                    errors += 1
                    continue
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'reminder_type': reminder_type,
                'applications_processed': len(applications),
                'reminders_sent': reminders_sent,
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Application reminders completed: {reminders_sent} sent, {errors} failed")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during application reminders: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Application reminders task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying reminders task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=90 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def send_application_reminder_email(self, application_id: int, reminder_type: str) -> Dict[str, Any]:
    """
    Send application reminder email for a specific application.
    
    Args:
        application_id: Application ID to send reminder for
        reminder_type: Type of reminder
        
    Returns:
        Dict with sending result
    """
    task_id = self.request.id
    logger.info(f"Sending application reminder for application {application_id}, task {task_id}")
    
    try:
        db = SessionLocal()
        
        try:
            # Get application with user and job details
            application = db.query(Application).join(User).join(Job).filter(
                Application.id == application_id
            ).first()
            
            if not application:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f'Application {application_id} not found',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate email content
            email_content = generate_application_reminder_email_content(
                application, 
                reminder_type
            )
            
            # Send email
            email_result = send_email(
                to_email=application.user.email,
                subject=email_content['subject'],
                html_content=email_content['html'],
                text_content=email_content['text']
            )
            
            if email_result['success']:
                # Update application follow-up count
                application.follow_up_count += 1
                application.last_follow_up_date = datetime.now()
                db.commit()
                
                result = {
                    'task_id': task_id,
                    'application_id': application_id,
                    'status': 'sent',
                    'reminder_type': reminder_type,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'task_id': task_id,
                    'application_id': application_id,
                    'status': 'failed',
                    'error': email_result.get('error', 'Email sending failed'),
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"Application reminder result for application {application_id}: {result['status']}")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during reminder email: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Application reminder email task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying reminder email task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'application_id': application_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=3, default_retry_delay=240)
def send_career_updates(self, update_type: str = 'weekly') -> Dict[str, Any]:
    """
    Send career guidance updates and insights to users.
    
    Args:
        update_type: Type of update (weekly, monthly, industry_trends)
        
    Returns:
        Dict with update results
    """
    task_id = self.request.id
    logger.info(f"Starting career updates task {task_id}, type: {update_type}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Preparing career updates'})
        
        db = SessionLocal()
        
        try:
            # Get users who want career updates
            if update_type == 'weekly':
                users = db.query(User).filter(
                    User.is_active == True,
                    User.email_notifications == True,
                    User.job_alert_frequency.in_(['daily', 'weekly'])
                ).all()
            elif update_type == 'monthly':
                users = db.query(User).filter(
                    User.is_active == True,
                    User.email_notifications == True
                ).all()
            else:  # industry_trends
                users = db.query(User).filter(
                    User.is_active == True,
                    User.email_notifications == True,
                    User.profile_completion_score >= 70  # Only for complete profiles
                ).all()
            
            logger.info(f"Sending {update_type} career updates to {len(users)} users")
            
            sent_count = 0
            error_count = 0
            
            # Update task state
            self.update_state(state='PROGRESS', meta={
                'status': 'Sending career updates',
                'users_to_process': len(users)
            })
            
            for idx, user in enumerate(users):
                try:
                    # Update progress
                    if idx % 10 == 0:
                        self.update_state(state='PROGRESS', meta={
                            'status': f'Processing user {idx + 1} of {len(users)}',
                            'sent_count': sent_count,
                            'error_count': error_count
                        })
                    
                    # Send career update email
                    update_result = send_career_update_email.delay(
                        user.id,
                        update_type
                    )
                    
                    # Wait for result
                    try:
                        result = update_result.get(timeout=120)
                        if result['status'] == 'sent':
                            sent_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error getting update result for user {user.id}: {str(e)}")
                        error_count += 1
                
                except Exception as e:
                    logger.error(f"Error processing career update for user {user.id}: {str(e)}")
                    error_count += 1
                    continue
            
            result = {
                'task_id': task_id,
                'status': 'completed',
                'update_type': update_type,
                'users_processed': len(users),
                'updates_sent': sent_count,
                'errors': error_count,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Career updates completed: {sent_count} sent, {error_count} failed")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during career updates: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Career updates task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying career updates task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def send_career_update_email(self, user_id: int, update_type: str) -> Dict[str, Any]:
    """
    Send career update email to a specific user.
    
    Args:
        user_id: User ID to send update to
        update_type: Type of career update
        
    Returns:
        Dict with sending result
    """
    task_id = self.request.id
    logger.info(f"Sending career update to user {user_id}, task {task_id}")
    
    try:
        db = SessionLocal()
        
        try:
            # Get user
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {
                    'task_id': task_id,
                    'status': 'failed',
                    'error': f'User {user_id} not found',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate personalized career content
            career_content = generate_career_update_content(user, update_type, db)
            
            # Generate email content
            email_content = generate_career_update_email_content(
                user, 
                career_content, 
                update_type
            )
            
            # Send email
            email_result = send_email(
                to_email=user.email,
                subject=email_content['subject'],
                html_content=email_content['html'],
                text_content=email_content['text']
            )
            
            if email_result['success']:
                result = {
                    'task_id': task_id,
                    'user_id': user_id,
                    'status': 'sent',
                    'update_type': update_type,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'task_id': task_id,
                    'user_id': user_id,
                    'status': 'failed',
                    'error': email_result.get('error', 'Email sending failed'),
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"Career update result for user {user_id}: {result['status']}")
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during career update email: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Career update email task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying career update email task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        
        # Final failure
        return {
            'task_id': task_id,
            'user_id': user_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def send_immediate_job_alert(self, user_id: int, job_id: int) -> Dict[str, Any]:
    """
    Send immediate job alert for high-match jobs.
    
    Args:
        user_id: User ID to send alert to
        job_id: Job ID for the alert
        
    Returns:
        Dict with sending result
    """
    task_id = self.request.id
    logger.info(f"Sending immediate job alert to user {user_id} for job {job_id}")
    
    try:
        db = SessionLocal()
        
        try:
            # Get user and job
            user = db.query(User).filter(User.id == user_id).first()
            job = db.query(Job).filter(Job.id == job_id).first()
            
            if not user or not job:
                return {
                    'task_id': task_id,  
                    'status': 'failed',
                    'error': 'User or job not found',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check if user wants immediate alerts
            if user.job_alert_frequency != 'immediate':
                return {
                    'task_id': task_id,
                    'status': 'skipped',
                    'reason': 'User does not want immediate alerts',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Send immediate alert email
            alert_result = send_job_alert_email.delay(user_id, [job_id], 'immediate')
            result = alert_result.get(timeout=60)
            
            return result
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during immediate alert: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as exc:
        logger.error(f"Immediate job alert task {task_id} failed: {str(exc)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30)
        
        return {
            'task_id': task_id,
            'status': 'failed',
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }


# Helper functions

def filter_jobs_for_user(user: User, jobs: List[Job]) -> List[Job]:
    """Filter jobs based on user preferences and profile."""
    relevant_jobs = []
    
    for job in jobs:
        # Skip if job doesn't meet basic criteria
        if not job.is_active:
            continue
            
        # Location matching
        if user.preferred_locations:
            location_match = any(
                pref_loc.lower() in job.location.lower() 
                for pref_loc in user.preferred_locations
            )
            if not location_match and not user.willing_to_relocate and not job.is_remote:
                continue
        
        # Industry matching
        if user.preferred_industries and job.industry:
            industry_match = any(
                pref_industry.lower() in job.industry.lower() 
                for pref_industry in user.preferred_industries
            )
            if not industry_match:
                continue
        
        # Job title matching
        if user.preferred_job_titles:
            title_match = any(
                pref_title.lower() in job.title.lower() 
                for pref_title in user.preferred_job_titles
            )
            if not title_match:
                # Allow if job has high veteran match score
                if job.veteran_match_score < 75:
                    continue
        
        # Salary matching
        if user.expected_salary_min and job.salary_max:
            if job.salary_max < user.expected_salary_min:
                continue
        
        if user.expected_salary_max and job.salary_min:
            if job.salary_min > user.expected_salary_max:
                continue
        
        # Must have decent veteran match score
        if job.veteran_match_score < 50:
            continue
        
        relevant_jobs.append(job)
    
    # Sort by veteran match score and limit to top 10
    relevant_jobs.sort(key=lambda x: x.veteran_match_score, reverse=True)
    return relevant_jobs[:10]


def generate_job_alert_email_content(user: User, jobs: List[Job], alert_type: str) -> Dict[str, str]:
    """Generate email content for job alerts."""
    # Email subject
    if alert_type == 'immediate':
        subject = f"🚨 High-Match Job Alert for {user.get_full_name()}"
    elif alert_type == 'daily':
        subject = f"📧 Daily Job Recommendations for {user.get_full_name()}"
    else:
        subject = f"💼 New Job Opportunities for {user.get_full_name()}"
    
    # HTML email template
    html_template = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
            .job-item { border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }
            .job-title { font-size: 18px; font-weight: bold; color: #2c3e50; }
            .company { font-size: 16px; color: #7f8c8d; }
            .location { color: #27ae60; }
            .match-score { background-color: #3498db; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
            .veteran-badge { background-color: #e74c3c; color: white; padding
