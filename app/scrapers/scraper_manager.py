import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.database import get_db, SessionLocal
from app.models.job import Job
from app.models.user import User
from app.scrapers.base_scraper import BaseScraper
from app.scrapers.job_portal_scraper import JobPortalScraper
from app.scrapers.psu_scraper import PSUScraper
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScrapingTask:
    """Data class for scraping task configuration."""
    scraper_name: str
    scraper_type: str
    enabled: bool = True
    schedule_interval: int = 3600  # seconds
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    max_jobs_per_run: int = 100
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.next_run is None:
            self.next_run = datetime.now()


@dataclass
class ScrapingResult:
    """Data class for scraping operation results."""
    task_name: str
    success: bool
    jobs_found: int = 0
    jobs_saved: int = 0
    jobs_updated: int = 0
    jobs_duplicated: int = 0
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class JobDeduplicator:
    """Handle job deduplication logic."""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        
    def generate_job_hash(self, job_data: Dict[str, Any]) -> str:
        """Generate unique hash for job based on key fields."""
        # Use title, company, and location for hash generation
        key_fields = [
            str(job_data.get('title', '')).lower().strip(),
            str(job_data.get('company', '')).lower().strip(),
            str(job_data.get('location', '')).lower().strip(),
            str(job_data.get('source_portal', '')).lower().strip()
        ]
        
        # Create hash from concatenated key fields
        hash_input = '|'.join(key_fields)
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, job_data: Dict[str, Any]) -> bool:
        """Check if job is duplicate in current batch."""
        job_hash = self.generate_job_hash(job_data)
        
        if job_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(job_hash)
        return False
    
    def check_database_duplicate(self, db: Session, job_data: Dict[str, Any]) -> Optional[Job]:
        """Check if similar job exists in database."""
        title = job_data.get('title', '').strip()
        company = job_data.get('company', '').strip()
        location = job_data.get('location', '').strip()
        
        if not title or not company:
            return None
        
        # Query for similar jobs (exact match on title and company)
        existing_job = db.query(Job).filter(
            and_(
                Job.title.ilike(f"%{title}%"),
                Job.company_name.ilike(f"%{company}%"),
                Job.is_active == True
            )
        ).first()
        
        return existing_job
    
    def clear_session(self):
        """Clear current session hashes."""
        self.seen_hashes.clear()


class ScraperManager:
    """Coordinate multiple scrapers, handle scheduling and data management."""
    
    def __init__(self):
        self.scrapers: Dict[str, BaseScraper] = {}
        self.tasks: Dict[str, ScrapingTask] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.deduplicator = JobDeduplicator()
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_jobs_found': 0,
            'total_jobs_saved': 0,
            'last_run_time': None,
            'uptime_start': datetime.now()
        }
        
        # Initialize default scrapers and tasks
        self._initialize_default_tasks()
    
    def _initialize_default_tasks(self):
        """Initialize default scraping tasks."""
        # Job portal scraping tasks
        portal_task = ScrapingTask(
            scraper_name="job_portals",
            scraper_type="job_portal",
            schedule_interval=7200,  # 2 hours
            max_jobs_per_run=200,
            parameters={
                "keywords": "ex-servicemen OR military OR veteran OR government",
                "location": "india",
                "portals": ["naukri", "indeed", "monster", "foundit"],
                "max_pages": 3
            }
        )
        
        # PSU scraping tasks
        psu_task = ScrapingTask(
            scraper_name="psu_organizations",
            scraper_type="psu",
            schedule_interval=14400,  # 4 hours
            max_jobs_per_run=100,
            parameters={
                "organizations": ["hal", "bel", "drdo", "isro", "ongc", "ntpc", "bhel", "gail"],
                "include_ex_servicemen_only": True,
                "max_jobs_per_org": 15
            }
        )
        
        # Veteran-specific job search task
        veteran_jobs_task = ScrapingTask(
            scraper_name="veteran_jobs",
            scraper_type="job_portal",
            schedule_interval=3600,  # 1 hour
            max_jobs_per_run=50,
            parameters={
                "keywords": "ex-servicemen OR ex-service OR military background OR defence personnel",
                "location": "delhi OR mumbai OR bangalore OR chennai OR hyderabad",
                "portals": ["naukri", "indeed"],
                "max_pages": 2
            }
        )
        
        self.tasks = {
            "job_portals": portal_task,
            "psu_organizations": psu_task,
            "veteran_jobs": veteran_jobs_task
        }
    
    async def initialize_scrapers(self):
        """Initialize all scraper instances."""
        try:
            # Initialize job portal scraper
            job_portal_scraper = JobPortalScraper()
            await job_portal_scraper.start()
            self.scrapers["job_portal"] = job_portal_scraper
            logger.info("Job portal scraper initialized")
            
            # Initialize PSU scraper
            psu_scraper = PSUScraper()
            await psu_scraper.start()
            self.scrapers["psu"] = psu_scraper
            logger.info("PSU scraper initialized")
            
            logger.info(f"Initialized {len(self.scrapers)} scrapers")
            
        except Exception as e:
            logger.error(f"Error initializing scrapers: {str(e)}")
            raise
    
    async def cleanup_scrapers(self):
        """Clean up scraper resources."""
        for scraper_name, scraper in self.scrapers.items():
            try:
                await scraper.stop()
                logger.info(f"Scraper {scraper_name} stopped")
            except Exception as e:
                logger.error(f"Error stopping scraper {scraper_name}: {str(e)}")
        
        self.scrapers.clear()
        
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def start_scheduler(self):
        """Start the scraping scheduler."""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        logger.info("Starting scraper scheduler")
        
        try:
            await self.initialize_scrapers()
            
            while self.running:
                # Check for tasks that need to run
                for task_name, task in self.tasks.items():
                    if not task.enabled:
                        continue
                    
                    if self._should_run_task(task):
                        logger.info(f"Running task: {task_name}")
                        
                        try:
                            result = await self._execute_task(task)
                            self._update_task_schedule(task)
                            self._update_stats(result)
                            
                            logger.info(
                                f"Task {task_name} completed: "
                                f"{result.jobs_found} found, {result.jobs_saved} saved, "
                                f"{result.jobs_duplicated} duplicates"
                            )
                            
                        except Exception as e:
                            logger.error(f"Error executing task {task_name}: {str(e)}")
                            self.stats['failed_runs'] += 1
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
        finally:
            await self.cleanup_scrapers()
    
    async def stop_scheduler(self):
        """Stop the scraping scheduler."""
        logger.info("Stopping scraper scheduler")
        self.running = False
        await self.cleanup_scrapers()
    
    def _should_run_task(self, task: ScrapingTask) -> bool:
        """Check if task should run based on schedule."""
        if not task.next_run:
            return True
        
        return datetime.now() >= task.next_run
    
    def _update_task_schedule(self, task: ScrapingTask):
        """Update task scheduling information."""
        task.last_run = datetime.now()
        task.next_run = datetime.now() + timedelta(seconds=task.schedule_interval)
    
    async def _execute_task(self, task: ScrapingTask) -> ScrapingResult:
        """Execute a single scraping task."""
        start_time = datetime.now()
        result = ScrapingResult(task_name=task.scraper_name, success=False)
        
        try:
            # Get appropriate scraper
            scraper = self.scrapers.get(task.scraper_type)
            if not scraper:
                raise ValueError(f"Scraper type {task.scraper_type} not found")
            
            # Clear deduplicator for new session
            self.deduplicator.clear_session()
            
            # Execute scraping
            jobs_data = await scraper.scrape_jobs(**task.parameters)
            result.jobs_found = len(jobs_data)
            
            # Process and save jobs to database
            save_result = await self._save_jobs_to_database(jobs_data)
            result.jobs_saved = save_result['saved']
            result.jobs_updated = save_result['updated']
            result.jobs_duplicated = save_result['duplicated']
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Task execution error: {str(e)}")
            result.errors.append(str(e))
        
        result.duration = (datetime.now() - start_time).total_seconds()
        return result
    
    async def _save_jobs_to_database(self, jobs_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Save scraped jobs to database with deduplication."""
        saved_count = 0
        updated_count = 0
        duplicate_count = 0
        
        db = SessionLocal()
        
        try:
            for job_data in jobs_data:
                try:
                    # Check for duplicates in current batch
                    if self.deduplicator.is_duplicate(job_data):
                        duplicate_count += 1
                        continue
                    
                    # Check for existing job in database
                    existing_job = self.deduplicator.check_database_duplicate(db, job_data)
                    
                    if existing_job:
                        # Update existing job with new information
                        if await self._update_job(db, existing_job, job_data):
                            updated_count += 1
                        else:
                            duplicate_count += 1
                    else:
                        # Create new job
                        if await self._create_job(db, job_data):
                            saved_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing job: {str(e)}")
                    continue
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database error: {str(e)}")
        finally:
            db.close()
        
        return {
            'saved': saved_count,
            'updated': updated_count,
            'duplicated': duplicate_count
        }
    
    async def _create_job(self, db: Session, job_data: Dict[str, Any]) -> bool:
        """Create new job record in database."""
        try:
            # Map scraped data to Job model fields
            job = Job(
                title=job_data.get('title', ''),
                company_name=job_data.get('company', ''),
                location=job_data.get('location', ''),
                description=job_data.get('description', ''),
                requirements=job_data.get('requirements', ''),
                experience_level=self._normalize_experience(job_data.get('experience', '')),
                job_type=self._normalize_job_type(job_data.get('job_type', 'full_time')),
                source_portal=job_data.get('source_portal', ''),
                source_url=job_data.get('job_url', ''),
                external_job_id=job_data.get('external_id', ''),
                posted_date=self._parse_date(job_data.get('posted_date')),
                scraped_at=datetime.now(),
                
                # Veteran-specific fields
                veteran_preference=job_data.get('is_veteran_friendly', False),
                veteran_match_score=job_data.get('veteran_match_score', 0.0),
                government_job=job_data.get('is_government_job', False),
                psu_job=job_data.get('is_psu_job', False),
                
                # Skills and requirements
                required_skills=job_data.get('skills', []),
                
                # Salary information
                salary_min=self._parse_salary(job_data.get('salary', ''), 'min'),
                salary_max=self._parse_salary(job_data.get('salary', ''), 'max'),
                
                # Default values
                is_active=True,
                view_count=0
            )
            
            # Set additional fields if available
            if job_data.get('qualification'):
                job.education_requirements = [job_data['qualification']]
            
            if job_data.get('organization_full_name'):
                job.company_name = job_data['organization_full_name']
                
            db.add(job)
            return True
            
        except Exception as e:
            logger.error(f"Error creating job: {str(e)}")
            return False
    
    async def _update_job(self, db: Session, existing_job: Job, job_data: Dict[str, Any]) -> bool:
        """Update existing job with new information."""
        try:
            updated = False
            
            # Update fields that might have changed
            if job_data.get('description') and not existing_job.description:
                existing_job.description = job_data['description']
                updated = True
            
            if job_data.get('requirements') and not existing_job.requirements:
                existing_job.requirements = job_data['requirements']
                updated = True
            
            if job_data.get('skills') and not existing_job.required_skills:
                existing_job.required_skills = job_data['skills']
                updated = True
            
            # Update veteran matching score if higher
            new_score = job_data.get('veteran_match_score', 0.0)
            if new_score > existing_job.veteran_match_score:
                existing_job.veteran_match_score = new_score
                updated = True
            
            # Update last seen timestamp
            existing_job.last_updated = datetime.now()
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating job: {str(e)}")
            return False
    
    def _normalize_experience(self, experience_text: str) -> str:
        """Normalize experience text to standard levels."""
        if not experience_text:
            return "not_specified"
        
        exp_lower = experience_text.lower()
        
        if any(word in exp_lower for word in ['fresher', '0 year', 'no experience']):
            return "entry"
        elif any(word in exp_lower for word in ['1-3', '2-4', '0-3']):
            return "entry"
        elif any(word in exp_lower for word in ['3-6', '4-7', '5-8']):
            return "mid"
        elif any(word in exp_lower for word in ['6+', '7+', '8+', 'senior']):
            return "senior"
        elif any(word in exp_lower for word in ['10+', '12+', 'executive', 'manager']):
            return "executive"
        else:
            return "mid"  # Default
    
    def _normalize_job_type(self, job_type_text: str) -> str:
        """Normalize job type to standard values."""
        if not job_type_text:
            return "full_time"
        
        job_type_lower = job_type_text.lower()
        
        if 'part' in job_type_lower:
            return "part_time"
        elif 'contract' in job_type_lower or 'temporary' in job_type_lower:
            return "contract"
        elif 'intern' in job_type_lower:
            return "internship"
        else:
            return "full_time"
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_text:
            return None
        
        try:
            # Try different date formats
            import dateutil.parser
            return dateutil.parser.parse(date_text)
        except:
            return None
    
    def _parse_salary(self, salary_text: str, part: str) -> Optional[int]:
        """Parse salary information from text."""
        if not salary_text or salary_text.lower() in ['not disclosed', 'not specified']:
            return None
        
        try:
            # Extract numbers from salary text
            import re
            numbers = re.findall(r'[\d,]+', salary_text.replace(',', ''))
            
            if not numbers:
                return None
            
            if len(numbers) == 1:
                # Single salary value
                return int(numbers[0]) if part == 'min' else int(numbers[0])
            elif len(numbers) >= 2:
                # Salary range
                return int(numbers[0]) if part == 'min' else int(numbers[1])
            
        except:
            pass
        
        return None
    
    def _update_stats(self, result: ScrapingResult):
        """Update manager statistics."""
        self.stats['total_runs'] += 1
        
        if result.success:
            self.stats['successful_runs'] += 1
        else:
            self.stats['failed_runs'] += 1
        
        self.stats['total_jobs_found'] += result.jobs_found
        self.stats['total_jobs_saved'] += result.jobs_saved
        self.stats['last_run_time'] = result.timestamp
    
    async def run_task_manually(self, task_name: str) -> ScrapingResult:
        """Manually run a specific task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")
        
        task = self.tasks[task_name]
        
        if not self.scrapers:
            await self.initialize_scrapers()
        
        result = await self._execute_task(task)
        self._update_stats(result)
        
        return result
    
    def add_task(self, task_name: str, task: ScrapingTask):
        """Add a new scraping task."""
        self.tasks[task_name] = task
        logger.info(f"Added new task: {task_name}")
    
    def remove_task(self, task_name: str):
        """Remove a scraping task."""
        if task_name in self.tasks:
            del self.tasks[task_name]
            logger.info(f"Removed task: {task_name}")
    
    def enable_task(self, task_name: str):
        """Enable a scraping task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            logger.info(f"Enabled task: {task_name}")
    
    def disable_task(self, task_name: str):
        """Disable a scraping task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            logger.info(f"Disabled task: {task_name}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        status = {}
        
        for task_name, task in self.tasks.items():
            status[task_name] = {
                'enabled': task.enabled,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'schedule_interval': task.schedule_interval,
                'parameters': task.parameters
            }
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper manager statistics."""
        uptime = datetime.now() - self.stats['uptime_start']
        
        return {
            'running': self.running,
            'uptime': str(uptime),
            'total_runs': self.stats['total_runs'],
            'successful_runs': self.stats['successful_runs'],
            'failed_runs': self.stats['failed_runs'],
            'success_rate': (self.stats['successful_runs'] / max(self.stats['total_runs'], 1)) * 100,
            'total_jobs_found': self.stats['total_jobs_found'],
            'total_jobs_saved': self.stats['total_jobs_saved'],
            'last_run_time': self.stats['last_run_time'].isoformat() if self.stats['last_run_time'] else None,
            'active_tasks': len([t for t in self.tasks.values() if t.enabled]),
            'total_tasks': len(self.tasks),
            'scrapers_initialized': len(self.scrapers)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the scraper manager."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'manager_running': self.running,
            'scrapers_count': len(self.scrapers),
            'active_tasks': len([t for t in self.tasks.values() if t.enabled]),
            'issues': []
        }
        
        # Check scraper health
        scraper_health = {}
        for name, scraper in self.scrapers.items():
            try:
                scraper_status = await scraper.health_check()
                scraper_health[name] = scraper_status['status']
                
                if scraper_status['status'] != 'healthy':
                    health['issues'].append(f"Scraper {name} is {scraper_status['status']}")
                    
            except Exception as e:
                scraper_health[name] = 'error'
                health['issues'].append(f"Scraper {name} health check failed: {str(e)}")
        
        health['scraper_health'] = scraper_health
        
        # Set overall status based on issues
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
        
        return health


# Global scraper manager instance
scraper_manager = ScraperManager()


async def start_scraper_manager():
    """Start the global scraper manager."""
    await scraper_manager.start_scheduler()


async def stop_scraper_manager():
    """Stop the global scraper manager."""
    await scraper_manager.stop_scheduler()


def get_scraper_manager() -> ScraperManager:
    """Get the global scraper manager instance."""
    return scraper_manager
