from celery import Celery
from celery import signals
from kombu import Queue
import os
from app.config import settings

# Create Celery application instance
celery_app = Celery(
    "veterancareer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'app.tasks.scraping_tasks',
        'app.tasks.matching_tasks',
        'app.tasks.notification_tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'app.tasks.scraping_tasks.*': {'queue': 'scraping'},
        'app.tasks.matching_tasks.*': {'queue': 'matching'},
        'app.tasks.notification_tasks.*': {'queue': 'notifications'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('scraping', routing_key='scraping'),
        Queue('matching', routing_key='matching'),
        Queue('notifications', routing_key='notifications'),
    ),
    
    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Worker settings
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'run-job-scraping': {
            'task': 'app.tasks.scraping_tasks.run_job_scraping',
            'schedule': 7200.0,  # Every 2 hours
        },
        'run-psu-scraping': {
            'task': 'app.tasks.scraping_tasks.run_psu_scraping',
            'schedule': 14400.0,  # Every 4 hours
        },
        'update-job-matches': {
            'task': 'app.tasks.matching_tasks.update_user_job_matches',
            'schedule': 3600.0,  # Every hour
        },
        'send-daily-job-alerts': {
            'task': 'app.tasks.notification_tasks.send_daily_job_alerts',
            'schedule': 86400.0,  # Daily
        },
        'cleanup-expired-jobs': {
            'task': 'app.tasks.scraping_tasks.cleanup_expired_jobs',
            'schedule': 43200.0,  # Every 12 hours
        },
    },
    beat_schedule_filename='celerybeat-schedule',
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Auto-discover tasks
celery_app.autodiscover_tasks([
    'app.tasks.scraping_tasks',
    'app.tasks.matching_tasks', 
    'app.tasks.notification_tasks'
])

# Configure logging
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery setup."""
    print(f'Request: {self.request!r}')
    return {'status': 'success', 'message': 'Celery is working!'}

# Health check task
@celery_app.task
def health_check():
    """Health check task to verify Celery is running."""
    return {
        'status': 'healthy',
        'timestamp': str(os.popen('date').read().strip()),
        'worker_id': os.getpid(),
        'broker_url': settings.celery_broker_url,
        'backend_url': settings.celery_result_backend
    }

# Task to test Redis connection
@celery_app.task
def test_redis_connection():
    """Test Redis connection."""
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        return {'status': 'success', 'message': 'Redis connection successful'}
    except Exception as e:
        return {'status': 'error', 'message': f'Redis connection failed: {str(e)}'}

# Function to get Celery app instance (for FastAPI integration)
def get_celery_app():
    """Get Celery application instance."""
    return celery_app

# Signal handlers for application lifecycle
@signals.worker_ready.connect
def worker_ready(sender=None, **kwargs):
    """Signal handler for when worker is ready."""
    print(f"Worker {sender} is ready")

@signals.worker_shutdown.connect
def worker_shutdown(sender=None, **kwargs):
    """Signal handler for when worker is shutting down."""
    print(f"Worker {sender} is shutting down")

@signals.task_prerun.connect
def task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Signal handler for before task execution."""
    print(f"Task {task.name}[{task_id}] starting")

@signals.task_postrun.connect
def task_postrun(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Signal handler for after task execution."""
    print(f"Task {task.name}[{task_id}] completed with state: {state}")

# Error handling
@signals.task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Handle task failures."""
    print(f"Task {sender.name}[{task_id}] failed: {exception}")
    # Here you could add custom error handling, logging, or notifications

# Custom task base class for common functionality
class BaseTask(celery_app.Task):
    """Base task class with common functionality."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        print(f"Task {self.name}[{task_id}] failed: {exc}")
        # Add custom failure handling here
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        print(f"Task {self.name}[{task_id}] retrying: {exc}")
        # Add custom retry handling here
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        print(f"Task {self.name}[{task_id}] succeeded")
        # Add custom success handling here

# Set the base task class
celery_app.Task = BaseTask

if __name__ == '__main__':
    celery_app.start()