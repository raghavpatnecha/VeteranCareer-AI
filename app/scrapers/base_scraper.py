import asyncio
import time
import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import aiohttp
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from bs4 import BeautifulSoup
import json
import os
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Data class for scraping results."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[datetime] = None
    response_time: Optional[float] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 30
    requests_per_hour: int = 500
    delay_between_requests: float = 2.0
    random_delay_range: tuple = (1.0, 3.0)
    backoff_multiplier: float = 2.0
    max_backoff_delay: float = 60.0


class RateLimiter:
    """Rate limiter for web scraping requests."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_timestamps = []
        self.last_request_time = 0
        self.consecutive_errors = 0
        
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        current_time = time.time()
        
        # Clean old timestamps (older than 1 hour)
        hour_ago = current_time - 3600
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > hour_ago]
        
        # Check hourly limit
        if len(self.request_timestamps) >= self.config.requests_per_hour:
            sleep_time = 3600 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Hourly rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Check per-minute limit (last 60 seconds)
        minute_ago = current_time - 60
        recent_requests = [ts for ts in self.request_timestamps if ts > minute_ago]
        
        if len(recent_requests) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - recent_requests[0])
            if sleep_time > 0:
                logger.warning(f"Per-minute rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Apply minimum delay between requests
        time_since_last = current_time - self.last_request_time
        min_delay = self.config.delay_between_requests
        
        # Add exponential backoff for consecutive errors
        if self.consecutive_errors > 0:
            backoff_delay = min(
                self.config.delay_between_requests * (self.config.backoff_multiplier ** self.consecutive_errors),
                self.config.max_backoff_delay
            )
            min_delay = max(min_delay, backoff_delay)
        
        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            
            # Add random jitter
            if self.config.random_delay_range:
                jitter = random.uniform(*self.config.random_delay_range)
                sleep_time += jitter
            
            await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_timestamps.append(time.time())
        self.last_request_time = time.time()
    
    def record_success(self):
        """Record successful request to reset error counter."""
        self.consecutive_errors = 0
    
    def record_error(self):
        """Record failed request for backoff calculation."""
        self.consecutive_errors += 1


class BaseScraper(ABC):
    """Base class for all web scrapers with common functionality."""
    
    def __init__(
        self,
        name: str,
        base_url: str,
        rate_limit_config: Optional[RateLimitConfig] = None,
        browser_config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.browser_config = browser_config or self._get_default_browser_config()
        
        # Browser and context management
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': None,
            'last_request_time': None
        }
        
        # Error tracking
        self.recent_errors = []
        self.max_error_history = 100
        
    def _get_default_browser_config(self) -> Dict[str, Any]:
        """Get default browser configuration."""
        return {
            'headless': True,
            'args': [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-default-apps',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-translate',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ],
            'ignore_default_args': ['--enable-automation'],
            'user_agent': settings.user_agent or 'VeteranCareer-Bot/1.0'
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Initialize browser and session."""
        try:
            self.stats['start_time'] = datetime.now()
            
            # Initialize Playwright
            self.playwright = await async_playwright().start()
            
            # Launch browser
            self.browser = await self.playwright.chromium.launch(**self.browser_config)
            
            # Create browser context
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=self.browser_config.get('user_agent', 'VeteranCareer-Bot/1.0')
            )
            
            # Initialize HTTP session
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': self.browser_config.get('user_agent', 'VeteranCareer-Bot/1.0'),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            logger.info(f"Scraper '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scraper '{self.name}': {str(e)}")
            await self.stop()
            raise
    
    async def stop(self):
        """Clean up browser and session resources."""
        try:
            if self.context:
                await self.context.close()
                self.context = None
            
            if self.browser:
                await self.browser.close()
                self.browser = None
            
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
            
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
            
            logger.info(f"Scraper '{self.name}' stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping scraper '{self.name}': {str(e)}")
    
    async def create_page(self, **kwargs) -> Page:
        """Create a new browser page with common settings."""
        if not self.context:
            raise RuntimeError("Scraper not initialized. Call start() first.")
        
        page = await self.context.new_page()
        
        # Set additional page settings
        await page.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        })
        
        return page
    
    async def fetch_html(self, url: str, **kwargs) -> ScrapingResult:
        """Fetch HTML content using HTTP session."""
        start_time = time.time()
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            self.stats['requests_made'] += 1
            self.stats['last_request_time'] = datetime.now()
            
            if not self.session:
                raise RuntimeError("HTTP session not initialized")
            
            async with self.session.get(url, **kwargs) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    html_content = await response.text()
                    self.rate_limiter.record_success()
                    self.stats['successful_requests'] += 1
                    
                    return ScrapingResult(
                        success=True,
                        data={'html': html_content, 'status': response.status},
                        url=url,
                        timestamp=datetime.now(),
                        response_time=response_time
                    )
                else:
                    error_msg = f"HTTP {response.status}: {response.reason}"
                    self._record_error(url, error_msg)
                    
                    return ScrapingResult(
                        success=False,
                        error=error_msg,
                        url=url,
                        timestamp=datetime.now(),
                        response_time=response_time
                    )
        
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Request failed: {str(e)}"
            self._record_error(url, error_msg)
            
            return ScrapingResult(
                success=False,
                error=error_msg,
                url=url,
                timestamp=datetime.now(),
                response_time=response_time
            )
    
    async def fetch_with_browser(self, url: str, wait_for: Optional[str] = None, **kwargs) -> ScrapingResult:
        """Fetch content using browser automation."""
        start_time = time.time()
        page = None
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            self.stats['requests_made'] += 1
            self.stats['last_request_time'] = datetime.now()
            
            page = await self.create_page()
            
            # Navigate to page
            response = await page.goto(url, wait_until='networkidle', **kwargs)
            
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=30000)
            
            # Get page content
            html_content = await page.content()
            response_time = time.time() - start_time
            
            if response and response.status == 200:
                self.rate_limiter.record_success()
                self.stats['successful_requests'] += 1
                
                return ScrapingResult(
                    success=True,
                    data={'html': html_content, 'status': response.status},
                    url=url,
                    timestamp=datetime.now(),
                    response_time=response_time
                )
            else:
                status = response.status if response else 'Unknown'
                error_msg = f"Browser navigation failed with status: {status}"
                self._record_error(url, error_msg)
                
                return ScrapingResult(
                    success=False,
                    error=error_msg,
                    url=url,
                    timestamp=datetime.now(),
                    response_time=response_time
                )
        
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Browser request failed: {str(e)}"
            self._record_error(url, error_msg)
            
            return ScrapingResult(
                success=False,
                error=error_msg,
                url=url,
                timestamp=datetime.now(),
                response_time=response_time
            )
        
        finally:
            if page:
                await page.close()
    
    def parse_html(self, html_content: str, parser: str = 'html.parser') -> BeautifulSoup:
        """Parse HTML content using BeautifulSoup."""
        return BeautifulSoup(html_content, parser)
    
    def extract_text(self, soup: BeautifulSoup, selector: str, default: str = '') -> str:
        """Extract text from soup using CSS selector."""
        try:
            element = soup.select_one(selector)
            return element.get_text(strip=True) if element else default
        except Exception as e:
            logger.warning(f"Text extraction failed for selector '{selector}': {str(e)}")
            return default
    
    def extract_attribute(self, soup: BeautifulSoup, selector: str, attribute: str, default: str = '') -> str:
        """Extract attribute value from soup using CSS selector."""
        try:
            element = soup.select_one(selector)
            return element.get(attribute, default) if element else default
        except Exception as e:
            logger.warning(f"Attribute extraction failed for selector '{selector}', attribute '{attribute}': {str(e)}")
            return default
    
    def extract_multiple(self, soup: BeautifulSoup, selector: str) -> List[BeautifulSoup]:
        """Extract multiple elements from soup using CSS selector."""
        try:
            return soup.select(selector)
        except Exception as e:
            logger.warning(f"Multiple extraction failed for selector '{selector}': {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ''
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common unwanted characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')   # Zero-width space
        
        return text.strip()
    
    def build_absolute_url(self, relative_url: str) -> str:
        """Convert relative URL to absolute URL."""
        if not relative_url:
            return ''
        
        return urljoin(self.base_url, relative_url)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _record_error(self, url: str, error: str):
        """Record error for tracking and analysis."""
        self.rate_limiter.record_error()
        self.stats['failed_requests'] += 1
        
        error_record = {
            'url': url,
            'error': error,
            'timestamp': datetime.now(),
            'scraper': self.name
        }
        
        self.recent_errors.append(error_record)
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_error_history:
            self.recent_errors = self.recent_errors[-self.max_error_history:]
        
        logger.error(f"Scraper '{self.name}' error for URL '{url}': {error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics."""
        current_time = datetime.now()
        
        stats = self.stats.copy()
        
        if stats['start_time']:
            stats['uptime'] = str(current_time - stats['start_time'])
            
            if stats['requests_made'] > 0:
                elapsed_hours = (current_time - stats['start_time']).total_seconds() / 3600
                stats['requests_per_hour'] = stats['requests_made'] / max(elapsed_hours, 0.001)
                stats['success_rate'] = (stats['successful_requests'] / stats['requests_made']) * 100
            else:
                stats['requests_per_hour'] = 0
                stats['success_rate'] = 0
        
        stats['recent_errors_count'] = len(self.recent_errors)
        
        return stats
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self.recent_errors[-limit:] if limit else self.recent_errors
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the scraper."""
        health_status = {
            'scraper_name': self.name,
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'browser_ready': self.browser is not None and self.context is not None,
            'session_ready': self.session is not None and not self.session.closed,
            'stats': self.get_stats()
        }
        
        # Check if there are too many recent errors
        recent_error_threshold = 5
        recent_errors = len([e for e in self.recent_errors 
                           if e['timestamp'] > datetime.now() - timedelta(minutes=10)])
        
        if recent_errors >= recent_error_threshold:
            health_status['status'] = 'degraded'
            health_status['warning'] = f"High error rate: {recent_errors} errors in last 10 minutes"
        
        return health_status
    
    @abstractmethod
    async def scrape_jobs(self, **kwargs) -> List[Dict[str, Any]]:
        """Abstract method to scrape jobs. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def scrape_job_details(self, job_url: str) -> Dict[str, Any]:
        """Abstract method to scrape detailed job information. Must be implemented by subclasses."""
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', base_url='{self.base_url}')>"
