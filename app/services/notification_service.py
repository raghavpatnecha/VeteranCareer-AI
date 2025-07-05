from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import json
from abc import ABC, abstractmethod

from app.models.user import User
from app.models.job import Job
from app.models.application import Application
from app.config import settings

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Enumeration for notification types."""
    JOB_ALERT = "job_alert"
    APPLICATION_UPDATE = "application_update"
    CAREER_GUIDANCE = "career_guidance"
    INTERVIEW_REMINDER = "interview_reminder"
    FOLLOW_UP_REMINDER = "follow_up_reminder"
    OFFER_RECEIVED = "offer_received"
    PROFILE_COMPLETION = "profile_completion"
    SKILL_RECOMMENDATION = "skill_recommendation"
    INDUSTRY_UPDATE = "industry_update"
    SYSTEM_NOTIFICATION = "system_notification"


class NotificationChannel(str, Enum):
    """Enumeration for notification channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    PUSH = "push"


class NotificationPriority(str, Enum):
    """Enumeration for notification priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationTemplate:
    """Data class for notification templates."""
    template_id: str
    notification_type: NotificationType
    channel: NotificationChannel
    subject_template: str
    body_template: str
    html_template: Optional[str] = None
    variables: List[str] = None
    is_active: bool = True


@dataclass
class NotificationContent:
    """Data class for notification content."""
    subject: str
    body: str
    html_content: Optional[str] = None
    variables: Dict[str, Any] = None
    attachments: List[str] = None


@dataclass
class NotificationRequest:
    """Data class for notification requests."""
    user_id: int
    notification_type: NotificationType
    channel: NotificationChannel
    priority: NotificationPriority
    content: NotificationContent
    scheduled_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class NotificationResult:
    """Data class for notification results."""
    success: bool
    notification_id: Optional[str] = None
    channel: Optional[NotificationChannel] = None
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """Send notification through this provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    @abstractmethod
    def get_supported_channels(self) -> List[NotificationChannel]:
        """Get supported notification channels."""
        pass


class EmailProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(self):
        self.smtp_configured = bool(settings.smtp_host and settings.smtp_username)
    
    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """Send email notification."""
        try:
            if not self.is_available():
                return NotificationResult(
                    success=False,
                    error_message="Email provider not configured"
                )
            
            # TODO: Implement actual email sending logic
            # This would integrate with SMTP server or email service
            
            return NotificationResult(
                success=True,
                notification_id=f"email_{datetime.now().timestamp()}",
                channel=NotificationChannel.EMAIL,
                sent_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if email provider is configured."""
        return self.smtp_configured
    
    def get_supported_channels(self) -> List[NotificationChannel]:
        """Get supported channels."""
        return [NotificationChannel.EMAIL]


class SMSProvider(NotificationProvider):
    """SMS notification provider."""
    
    def __init__(self):
        self.sms_configured = False  # TODO: Configure SMS service
    
    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """Send SMS notification."""
        try:
            if not self.is_available():
                return NotificationResult(
                    success=False,
                    error_message="SMS provider not configured"
                )
            
            # TODO: Implement actual SMS sending logic
            # This would integrate with SMS service provider
            
            return NotificationResult(
                success=True,
                notification_id=f"sms_{datetime.now().timestamp()}",
                channel=NotificationChannel.SMS,
                sent_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {str(e)}")
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if SMS provider is configured."""
        return self.sms_configured
    
    def get_supported_channels(self) -> List[NotificationChannel]:
        """Get supported channels."""
        return [NotificationChannel.SMS]


class InAppProvider(NotificationProvider):
    """In-app notification provider."""
    
    async def send_notification(self, request: NotificationRequest) -> NotificationResult:
        """Send in-app notification."""
        try:
            # TODO: Implement in-app notification storage
            # This would store notifications in database for app display
            
            return NotificationResult(
                success=True,
                notification_id=f"inapp_{datetime.now().timestamp()}",
                channel=NotificationChannel.IN_APP,
                sent_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error sending in-app notification: {str(e)}")
            return NotificationResult(
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """In-app notifications are always available."""
        return True
    
    def get_supported_channels(self) -> List[NotificationChannel]:
        """Get supported channels."""
        return [NotificationChannel.IN_APP]


class NotificationService:
    """Main notification service for managing all notification types."""
    
    def __init__(self):
        # Initialize notification providers
        self.providers = {
            NotificationChannel.EMAIL: EmailProvider(),
            NotificationChannel.SMS: SMSProvider(),
            NotificationChannel.IN_APP: InAppProvider()
        }
        
        # Initialize notification templates
        self.templates = self._load_notification_templates()
        
        # Statistics tracking
        self.stats = {
            'notifications_sent': 0,
            'notifications_failed': 0,
            'last_sent_time': None
        }
    
    def _load_notification_templates(self) -> Dict[str, NotificationTemplate]:
        """Load notification templates."""
        templates = {}
        
        # Job Alert Templates
        templates['job_alert_email'] = NotificationTemplate(
            template_id='job_alert_email',
            notification_type=NotificationType.JOB_ALERT,
            channel=NotificationChannel.EMAIL,
            subject_template='🔔 New Job Opportunities for {user_name}',
            body_template='''Dear {user_name},

We found {job_count} new job opportunities that match your profile:

{job_list}

Visit VeteranCareer AI to view all opportunities and apply.

Best regards,
VeteranCareer AI Team''',
            html_template='''<h2>New Job Opportunities</h2>
<p>Dear {user_name},</p>
<p>We found {job_count} new job opportunities that match your profile:</p>
{job_list_html}
<p><a href="{dashboard_url}">View All Opportunities</a></p>''',
            variables=['user_name', 'job_count', 'job_list', 'job_list_html', 'dashboard_url']
        )
        
        # Application Update Templates
        templates['application_status_email'] = NotificationTemplate(
            template_id='application_status_email',
            notification_type=NotificationType.APPLICATION_UPDATE,
            channel=NotificationChannel.EMAIL,
            subject_template='📋 Application Update: {job_title} at {company_name}',
            body_template='''Dear {user_name},

Your application for {job_title} at {company_name} has been updated.

Status: {application_status}
{status_details}

View your application: {application_url}

Best regards,
VeteranCareer AI Team''',
            variables=['user_name', 'job_title', 'company_name', 'application_status', 'status_details', 'application_url']
        )
        
        # Career Guidance Templates
        templates['career_advice_email'] = NotificationTemplate(
            template_id='career_advice_email',
            notification_type=NotificationType.CAREER_GUIDANCE,
            channel=NotificationChannel.EMAIL,
            subject_template='💼 Weekly Career Insights for {user_name}',
            body_template='''Dear {user_name},

Here are your personalized career insights for this week:

{career_insights}

Industry Trends:
{industry_trends}

Recommended Actions:
{recommended_actions}

Best regards,
VeteranCareer AI Team''',
            variables=['user_name', 'career_insights', 'industry_trends', 'recommended_actions']
        )
        
        # Interview Reminder Templates
        templates['interview_reminder_email'] = NotificationTemplate(
            template_id='interview_reminder_email',
            notification_type=NotificationType.INTERVIEW_REMINDER,
            channel=NotificationChannel.EMAIL,
            subject_template='📅 Interview Reminder: {job_title} at {company_name}',
            body_template='''Dear {user_name},

This is a reminder about your upcoming interview:

Position: {job_title}
Company: {company_name}
Date & Time: {interview_date}
Type: {interview_type}
Location: {interview_location}

Preparation Tips:
{preparation_tips}

Good luck!
VeteranCareer AI Team''',
            variables=['user_name', 'job_title', 'company_name', 'interview_date', 'interview_type', 'interview_location', 'preparation_tips']
        )
        
        return templates
    
    async def send_job_alert(
        self,
        user: User,
        jobs: List[Job],
        alert_type: str = 'daily',
        channels: List[NotificationChannel] = None
    ) -> List[NotificationResult]:
        """Send job alert notification to user."""
        try:
            if not channels:
                channels = [NotificationChannel.EMAIL] if user.email_notifications else []
                if user.sms_notifications:
                    channels.append(NotificationChannel.SMS)
            
            # Prepare job list content
            job_list = []
            job_list_html = []
            
            for job in jobs[:10]:  # Limit to top 10 jobs
                job_item = f"• {job.title} at {job.company_name} - {job.location}"
                if job.salary_min or job.salary_max:
                    job_item += f" ({job.get_salary_range_display()})"
                job_list.append(job_item)
                
                job_html = f'''<div style="border: 1px solid #ddd; padding: 10px; margin: 5px 0;">
                    <h4>{job.title}</h4>
                    <p><strong>{job.company_name}</strong> - {job.location}</p>
                    <p>Match Score: {job.veteran_match_score}%</p>
                </div>'''
                job_list_html.append(job_html)
            
            # Prepare notification content
            content = NotificationContent(
                subject=f"🔔 {len(jobs)} New Job Opportunities",
                body=f"Found {len(jobs)} new job opportunities matching your profile",
                variables={
                    'user_name': user.get_full_name(),
                    'job_count': len(jobs),
                    'job_list': '\n'.join(job_list),
                    'job_list_html': '\n'.join(job_list_html),
                    'dashboard_url': f"{settings.app_name}/dashboard"
                }
            )
            
            # Send notifications through all channels
            results = []
            for channel in channels:
                request = NotificationRequest(
                    user_id=user.id,
                    notification_type=NotificationType.JOB_ALERT,
                    channel=channel,
                    priority=NotificationPriority.MEDIUM,
                    content=content,
                    metadata={'alert_type': alert_type, 'job_count': len(jobs)}
                )
                
                result = await self._send_notification(request)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending job alert to user {user.id}: {str(e)}")
            return [NotificationResult(success=False, error_message=str(e))]
    
    async def send_application_update(
        self,
        user: User,
        application: Application,
        update_type: str,
        channels: List[NotificationChannel] = None
    ) -> List[NotificationResult]:
        """Send application status update notification."""
        try:
            if not channels:
                channels = [NotificationChannel.EMAIL] if user.email_notifications else []
                channels.append(NotificationChannel.IN_APP)  # Always send in-app
            
            # Prepare status-specific content
            status_details = self._get_status_details(application, update_type)
            
            content = NotificationContent(
                subject=f"Application Update: {application.job.title}",
                body=f"Your application status has been updated to: {application.get_status_display()}",
                variables={
                    'user_name': user.get_full_name(),
                    'job_title': application.job.title,
                    'company_name': application.job.company_name,
                    'application_status': application.get_status_display(),
                    'status_details': status_details,
                    'application_url': f"{settings.app_name}/applications/{application.id}"
                }
            )
            
            # Send notifications
            results = []
            for channel in channels:
                request = NotificationRequest(
                    user_id=user.id,
                    notification_type=NotificationType.APPLICATION_UPDATE,
                    channel=channel,
                    priority=NotificationPriority.HIGH,
                    content=content,
                    metadata={'application_id': application.id, 'update_type': update_type}
                )
                
                result = await self._send_notification(request)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending application update to user {user.id}: {str(e)}")
            return [NotificationResult(success=False, error_message=str(e))]
    
    async def send_career_guidance(
        self,
        user: User,
        guidance_type: str,
        content_data: Dict[str, Any],
        channels: List[NotificationChannel] = None
    ) -> List[NotificationResult]:
        """Send career guidance notification."""
        try:
            if not channels:
                channels = [NotificationChannel.EMAIL] if user.email_notifications else []
            
            # Prepare guidance content based on type
            if guidance_type == 'weekly_insights':
                subject = f"💼 Weekly Career Insights"
                insights = content_data.get('insights', [])
                trends = content_data.get('trends', [])
                actions = content_data.get('recommended_actions', [])
                
                content = NotificationContent(
                    subject=subject,
                    body="Your weekly career insights are ready",
                    variables={
                        'user_name': user.get_full_name(),
                        'career_insights': '\n'.join(f"• {insight}" for insight in insights),
                        'industry_trends': '\n'.join(f"• {trend}" for trend in trends),
                        'recommended_actions': '\n'.join(f"• {action}" for action in actions)
                    }
                )
            
            elif guidance_type == 'skill_recommendation':
                subject = f"🎯 New Skill Recommendations"
                skills = content_data.get('recommended_skills', [])
                
                content = NotificationContent(
                    subject=subject,
                    body=f"We recommend {len(skills)} new skills for your career growth",
                    variables={
                        'user_name': user.get_full_name(),
                        'recommended_skills': '\n'.join(f"• {skill}" for skill in skills),
                        'profile_url': f"{settings.app_name}/profile"
                    }
                )
            
            else:
                # Generic career guidance
                content = NotificationContent(
                    subject=f"Career Guidance Update",
                    body=content_data.get('message', 'New career guidance available'),
                    variables={
                        'user_name': user.get_full_name(),
                        **content_data.get('variables', {})
                    }
                )
            
            # Send notifications
            results = []
            for channel in channels:
                request = NotificationRequest(
                    user_id=user.id,
                    notification_type=NotificationType.CAREER_GUIDANCE,
                    channel=channel,
                    priority=NotificationPriority.LOW,
                    content=content,
                    metadata={'guidance_type': guidance_type}
                )
                
                result = await self._send_notification(request)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending career guidance to user {user.id}: {str(e)}")
            return [NotificationResult(success=False, error_message=str(e))]
    
    async def send_interview_reminder(
        self,
        user: User,
        application: Application,
        reminder_hours: int = 24,
        channels: List[NotificationChannel] = None
    ) -> List[NotificationResult]:
        """Send interview reminder notification."""
        try:
            if not channels:
                channels = [NotificationChannel.EMAIL, NotificationChannel.IN_APP]
                if user.sms_notifications:
                    channels.append(NotificationChannel.SMS)
            
            # Prepare interview details
            interview_date = application.interview_scheduled_date.strftime("%B %d, %Y at %H:%M") if application.interview_scheduled_date else "TBD"
            preparation_tips = [
                "Research the company and role thoroughly",
                "Prepare examples from your military experience",
                "Practice common interview questions",
                "Prepare questions about the role and company",
                "Test your technology if it's a video interview"
            ]
            
            content = NotificationContent(
                subject=f"Interview Reminder: {application.job.title}",
                body=f"Interview reminder for {application.job.title} at {application.job.company_name}",
                variables={
                    'user_name': user.get_full_name(),
                    'job_title': application.job.title,
                    'company_name': application.job.company_name,
                    'interview_date': interview_date,
                    'interview_type': application.interview_type or 'Not specified',
                    'interview_location': application.interview_location or 'Not specified',
                    'preparation_tips': '\n'.join(f"• {tip}" for tip in preparation_tips)
                }
            )
            
            # Send notifications
            results = []
            for channel in channels:
                priority = NotificationPriority.HIGH if reminder_hours <= 2 else NotificationPriority.MEDIUM
                
                request = NotificationRequest(
                    user_id=user.id,
                    notification_type=NotificationType.INTERVIEW_REMINDER,
                    channel=channel,
                    priority=priority,
                    content=content,
                    metadata={'application_id': application.id, 'reminder_hours': reminder_hours}
                )
                
                result = await self._send_notification(request)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending interview reminder to user {user.id}: {str(e)}")
            return [NotificationResult(success=False, error_message=str(e))]
    
    async def send_follow_up_reminder(
        self,
        user: User,
        application: Application,
        reminder_type: str = 'status_check',
        channels: List[NotificationChannel] = None
    ) -> List[NotificationResult]:
        """Send follow-up reminder notification."""
        try:
            if not channels:
                channels = [NotificationChannel.EMAIL, NotificationChannel.IN_APP]
            
            # Customize message based on reminder type
            if reminder_type == 'thank_you':
                subject = f"Send Thank You Note: {application.job.title}"
                message = "Consider sending a thank you note after your interview"
            elif reminder_type == 'status_check':
                subject = f"Follow Up: {application.job.title}"
                message = "It's been a while since your application. Consider following up"
            else:
                subject = f"Application Reminder: {application.job.title}"
                message = "Application follow-up reminder"
            
            content = NotificationContent(
                subject=subject,
                body=message,
                variables={
                    'user_name': user.get_full_name(),
                    'job_title': application.job.title,
                    'company_name': application.job.company_name,
                    'application_status': application.get_status_display(),
                    'days_since_application': application.calculate_days_since_application()
                }
            )
            
            # Send notifications
            results = []
            for channel in channels:
                request = NotificationRequest(
                    user_id=user.id,
                    notification_type=NotificationType.FOLLOW_UP_REMINDER,
                    channel=channel,
                    priority=NotificationPriority.MEDIUM,
                    content=content,
                    metadata={'application_id': application.id, 'reminder_type': reminder_type}
                )
                
                result = await self._send_notification(request)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending follow-up reminder to user {user.id}: {str(e)}")
            return [NotificationResult(success=False, error_message=str(e))]
    
    async def send_offer_notification(
        self,
        user: User,
        application: Application,
        channels: List[NotificationChannel] = None
    ) -> List[NotificationResult]:
        """Send job offer notification."""
        try:
            if not channels:
                channels = [NotificationChannel.EMAIL, NotificationChannel.IN_APP]
                if user.sms_notifications:
                    channels.append(NotificationChannel.SMS)
            
            salary_info = ""
            if application.offered_salary:
                salary_info = f"Offered Salary: ₹{application.offered_salary:,}"
            
            deadline_info = ""
            if application.offer_deadline:
                deadline_info = f"Response Required By: {application.offer_deadline.strftime('%B %d, %Y')}"
            
            content = NotificationContent(
                subject=f"🎉 Job Offer Received: {application.job.title}",
                body=f"Congratulations! You received a job offer from {application.job.company_name}",
                variables={
                    'user_name': user.get_full_name(),
                    'job_title': application.job.title,
                    'company_name': application.job.company_name,
                    'salary_info': salary_info,
                    'deadline_info': deadline_info,
                    'offer_details': application.offer_details or "See application for details"
                }
            )
            
            # Send notifications
            results = []
            for channel in channels:
                request = NotificationRequest(
                    user_id=user.id,
                    notification_type=NotificationType.OFFER_RECEIVED,
                    channel=channel,
                    priority=NotificationPriority.HIGH,
                    content=content,
                    metadata={'application_id': application.id}
                )
                
                result = await self._send_notification(request)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending offer notification to user {user.id}: {str(e)}")
            return [NotificationResult(success=False, error_message=str(e))]
    
    async def _send_notification(self, request: NotificationRequest) -> NotificationResult:
        """Send notification through appropriate provider."""
        try:
            provider = self.providers.get(request.channel)
            if not provider or not provider.is_available():
                return NotificationResult(
                    success=False,
                    error_message=f"Provider for {request.channel} not available"
                )
            
            # Apply template if available
            template_key = f"{request.notification_type}_{request.channel}"
            if template_key in self.templates:
                template = self.templates[template_key]
                request.content = self._apply_template(template, request.content)
            
            # Send notification
            result = await provider.send_notification(request)
            
            # Update statistics
            if result.success:
                self.stats['notifications_sent'] += 1
                self.stats['last_sent_time'] = datetime.now()
            else:
                self.stats['notifications_failed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return NotificationResult(success=False, error_message=str(e))
    
    def _apply_template(self, template: NotificationTemplate, content: NotificationContent) -> NotificationContent:
        """Apply notification template to content."""
        try:
            variables = content.variables or {}
            
            # Apply template to subject
            subject = template.subject_template
            for var, value in variables.items():
                subject = subject.replace(f"{{{var}}}", str(value))
            
            # Apply template to body
            body = template.body_template
            for var, value in variables.items():
                body = body.replace(f"{{{var}}}", str(value))
            
            # Apply template to HTML if available
            html_content = None
            if template.html_template:
                html_content = template.html_template
                for var, value in variables.items():
                    html_content = html_content.replace(f"{{{var}}}", str(value))
            
            return NotificationContent(
                subject=subject,
                body=body,
                html_content=html_content,
                variables=variables,
                attachments=content.attachments
            )
            
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
            return content
    
    def _get_status_details(self, application: Application, update_type: str) -> str:
        """Get detailed status information for application updates."""
        status_details = {
            'applied': "Your application has been submitted successfully.",
            'viewed': "Your application has been reviewed by the employer.",
            'shortlisted': "Congratulations! You have been shortlisted for the next round.",
            'interview_scheduled': f"Interview scheduled for {application.interview_scheduled_date.strftime('%B %d, %Y') if application.interview_scheduled_date else 'TBD'}",
            'interview_completed': "Interview completed. Awaiting feedback.",
            'under_review': "Your application is under review by the hiring team.",
            'offered': "Congratulations! You have received a job offer.",
            'accepted': "Offer accepted. Welcome to your new role!",
            'rejected': "Unfortunately, your application was not selected.",
            'withdrawn': "Application has been withdrawn."
        }
        
        return status_details.get(application.status, "Application status updated.")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification service statistics."""
        return {
            **self.stats,
            'providers_status': {
                channel.value: provider.is_available() 
                for channel, provider in self.providers.items()
            },
            'templates_loaded': len(self.templates)
        }
    
    def get_user_notification_preferences(self, user: User) -> Dict[str, bool]:
        """Get user's notification preferences."""
        return {
            'email_notifications': user.email_notifications,
            'sms_notifications': user.sms_notifications,
            'job_alert_frequency': user.job_alert_frequency,
            'channels_enabled': [
                channel for channel, enabled in {
                    'email': user.email_notifications,
                    'sms': user.sms_notifications,
                    'in_app': True  # Always enabled
                }.items() if enabled
            ]
        }


# Global notification service instance
notification_service = NotificationService()


def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    return notification_service
