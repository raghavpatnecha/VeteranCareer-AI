import re
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
import json
import logging
from dataclasses import dataclass

from app.scrapers.base_scraper import BaseScraper, ScrapingResult, RateLimitConfig
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PSUJobListing:
    """Data class for PSU/Government job listing information."""
    title: str
    organization: str
    department: str
    location: str
    qualification: str
    experience: str
    age_limit: str
    vacancy_count: str
    application_deadline: str
    advertisement_number: str
    job_url: str = ""
    description: str = ""
    requirements: str = ""
    salary_range: str = ""
    reservation_details: str = ""
    application_fee: str = ""
    selection_process: str = ""
    is_ex_servicemen_eligible: bool = False
    ex_servicemen_age_relaxation: str = ""
    source_organization: str = ""
    posted_date: str = ""
    
    def __post_init__(self):
        if not self.source_organization:
            self.source_organization = self.organization


class PSUScraper(BaseScraper):
    """Scraper for PSU and government defense organizations like HAL, BEL, DRDO, ISRO."""
    
    def __init__(self, organization_name: str = "multi_psu"):
        # Configure rate limiting for government websites
        rate_config = RateLimitConfig(
            requests_per_minute=10,  # Conservative for government sites
            requests_per_hour=150,
            delay_between_requests=5.0,
            random_delay_range=(3.0, 8.0)
        )
        
        super().__init__(
            name=organization_name,
            base_url="https://www.hal-india.co.in",  # Default to HAL
            rate_limit_config=rate_config
        )
        
        # PSU/Government organization configurations
        self.psu_configs = {
            "hal": {
                "name": "Hindustan Aeronautics Limited",
                "base_url": "https://www.hal-india.co.in",
                "careers_url": "https://www.hal-india.co.in/careers",
                "job_listings_url": "https://www.hal-india.co.in/career/current-openings",
                "selectors": {
                    "job_cards": ".career-opening, .job-listing, .vacancy-item",
                    "title": ".job-title, .post-name, h3, h4",
                    "department": ".department, .division",
                    "location": ".location, .place",
                    "qualification": ".qualification, .eligibility",
                    "experience": ".experience, .exp-required",
                    "deadline": ".last-date, .deadline, .closing-date",
                    "vacancy_count": ".vacancy, .posts, .no-of-posts",
                    "advertisement_no": ".advt-no, .notification-no",
                    "details_url": "a[href]"
                }
            },
            "bel": {
                "name": "Bharat Electronics Limited",
                "base_url": "https://bel-india.in",
                "careers_url": "https://bel-india.in/careers.html",
                "job_listings_url": "https://bel-india.in/currentopenings.html",
                "selectors": {
                    "job_cards": ".job-opening, .career-item, .vacancy",
                    "title": ".designation, .post, h3",
                    "location": ".location, .unit",
                    "qualification": ".qualification, .educational-qualification",
                    "experience": ".experience",
                    "deadline": ".last-date, .closing-date",
                    "advertisement_no": ".advt-no, .notification",
                    "details_url": "a"
                }
            },
            "drdo": {
                "name": "Defence Research and Development Organisation",
                "base_url": "https://www.drdo.gov.in",
                "careers_url": "https://www.drdo.gov.in/careers",
                "job_listings_url": "https://www.drdo.gov.in/career-opportunities",
                "selectors": {
                    "job_cards": ".career-opening, .job-vacancy, .recruitment-item",
                    "title": ".post-title, .designation, h3",
                    "organization": ".organization, .lab-name",
                    "location": ".location, .place",
                    "qualification": ".qualification, .eligibility-criteria",
                    "deadline": ".last-date, .closing-date",
                    "advertisement_no": ".advt-no, .recruitment-no",
                    "details_url": "a[href]"
                }
            },
            "isro": {
                "name": "Indian Space Research Organisation",
                "base_url": "https://www.isro.gov.in",
                "careers_url": "https://www.isro.gov.in/careers",
                "job_listings_url": "https://www.isro.gov.in/careers/current-opportunities",
                "selectors": {
                    "job_cards": ".career-item, .job-opening, .vacancy-details",
                    "title": ".post-name, .designation, h3, h4",
                    "location": ".location, .centre",
                    "qualification": ".qualification, .educational-requirement",
                    "experience": ".experience, .exp-required",
                    "deadline": ".last-date, .application-deadline",
                    "advertisement_no": ".advt-no, .notification-no",
                    "details_url": "a"
                }
            },
            "ongc": {
                "name": "Oil and Natural Gas Corporation",
                "base_url": "https://www.ongcindia.com",
                "careers_url": "https://www.ongcindia.com/web/eng/careers",
                "job_listings_url": "https://www.ongcindia.com/web/eng/careers/current-openings",
                "selectors": {
                    "job_cards": ".career-opening, .job-item, .recruitment",
                    "title": ".post, .designation, h3",
                    "location": ".location, .work-location",
                    "qualification": ".qualification, .eligibility",
                    "deadline": ".last-date, .closing-date",
                    "advertisement_no": ".advt-no, .recruitment-no"
                }
            },
            "ntpc": {
                "name": "National Thermal Power Corporation",
                "base_url": "https://www.ntpc.co.in",
                "careers_url": "https://www.ntpc.co.in/careers",
                "job_listings_url": "https://www.ntpc.co.in/careers/current-openings",
                "selectors": {
                    "job_cards": ".career-item, .job-opening, .vacancy",
                    "title": ".designation, .post-title, h3",
                    "location": ".location, .station",
                    "qualification": ".qualification, .educational-criteria",
                    "deadline": ".last-date, .application-deadline"
                }
            },
            "bhel": {
                "name": "Bharat Heavy Electricals Limited",
                "base_url": "https://www.bhel.com",
                "careers_url": "https://www.bhel.com/careers",
                "job_listings_url": "https://www.bhel.com/careers/current-vacancies",
                "selectors": {
                    "job_cards": ".career-opening, .job-vacancy, .recruitment-item",
                    "title": ".post-name, .designation, h3",
                    "location": ".location, .unit",
                    "qualification": ".qualification, .eligibility",
                    "deadline": ".last-date, .closing-date"
                }
            },
            "gail": {
                "name": "Gas Authority of India Limited",
                "base_url": "https://www.gailonline.com",
                "careers_url": "https://www.gailonline.com/careers",
                "job_listings_url": "https://www.gailonline.com/careers/current-openings",
                "selectors": {
                    "job_cards": ".career-item, .job-opening",
                    "title": ".designation, .post, h3",
                    "location": ".location, .work-place",
                    "qualification": ".qualification",
                    "deadline": ".last-date"
                }
            }
        }
        
        # Keywords that indicate ex-servicemen eligibility
        self.ex_servicemen_keywords = [
            "ex-servicemen", "ex servicemen", "ex-service", "ex service",
            "retired defence personnel", "retired military", "army personnel",
            "navy personnel", "air force personnel", "defence background",
            "military background", "service personnel", "armed forces",
            "commissioned officers", "non-commissioned officers",
            "ex-officers", "retired officers", "ex-soldiers", "veterans",
            "defence services", "military services", "paramilitary",
            "central armed police forces", "capf"
        ]
        
        # Age relaxation keywords for ex-servicemen
        self.age_relaxation_keywords = [
            "age relaxation", "relaxation in age", "upper age limit relaxed",
            "age concession", "additional age", "extra years", "relaxed age",
            "defence personnel relaxation", "ex-servicemen relaxation"
        ]
        
        # Common government job categories relevant to ex-servicemen
        self.relevant_categories = [
            "security", "administration", "technical", "engineering",
            "operations", "maintenance", "project management", "logistics",
            "training", "quality", "safety", "procurement", "finance",
            "human resources", "officer", "executive", "assistant",
            "supervisor", "manager", "specialist", "consultant"
        ]
    
    async def scrape_jobs(
        self,
        organizations: List[str] = None,
        include_ex_servicemen_only: bool = True,
        max_jobs_per_org: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape jobs from PSU and government organizations.
        
        Args:
            organizations: List of organizations to scrape (default: all)
            include_ex_servicemen_only: Filter for ex-servicemen eligible jobs
            max_jobs_per_org: Maximum jobs to scrape per organization
            
        Returns:
            List of job dictionaries
        """
        if organizations is None:
            organizations = list(self.psu_configs.keys())
        
        all_jobs = []
        
        for org in organizations:
            if org not in self.psu_configs:
                logger.warning(f"Unknown organization: {org}")
                continue
                
            try:
                logger.info(f"Scraping jobs from {org.upper()}")
                org_jobs = await self._scrape_organization_jobs(org, max_jobs_per_org)
                
                # Add source organization to each job
                for job in org_jobs:
                    job["source_organization"] = org
                    job["organization_full_name"] = self.psu_configs[org]["name"]
                
                all_jobs.extend(org_jobs)
                logger.info(f"Found {len(org_jobs)} jobs from {org.upper()}")
                
            except Exception as e:
                logger.error(f"Error scraping {org}: {str(e)}")
                continue
        
        # Filter for ex-servicemen eligible jobs if requested
        if include_ex_servicemen_only:
            ex_servicemen_jobs = self._filter_ex_servicemen_jobs(all_jobs)
            logger.info(f"Total jobs found: {len(all_jobs)}, Ex-servicemen eligible: {len(ex_servicemen_jobs)}")
            return ex_servicemen_jobs
        
        logger.info(f"Total jobs found: {len(all_jobs)}")
        return all_jobs
    
    async def _scrape_organization_jobs(self, org: str, max_jobs: int) -> List[Dict[str, Any]]:
        """Scrape jobs from a specific organization."""
        config = self.psu_configs[org]
        jobs = []
        
        # Update base URL for the organization
        self.base_url = config["base_url"]
        
        try:
            # Try multiple URLs for job listings
            urls_to_try = [
                config.get("job_listings_url"),
                config.get("careers_url"),
                f"{config['base_url']}/careers",
                f"{config['base_url']}/recruitment",
                f"{config['base_url']}/current-openings"
            ]
            
            for url in urls_to_try:
                if not url:
                    continue
                    
                logger.info(f"Trying URL: {url}")
                
                # Fetch job listings page
                result = await self.fetch_with_browser(url, wait_for="body")
                
                if not result.success:
                    logger.warning(f"Failed to fetch {url}: {result.error}")
                    continue
                
                # Parse job listings from the page
                page_jobs = self._parse_psu_job_listings(result.data["html"], org)
                
                if page_jobs:
                    jobs.extend(page_jobs[:max_jobs])
                    logger.info(f"Found {len(page_jobs)} jobs from {url}")
                    break  # Success, no need to try other URLs
                else:
                    logger.info(f"No jobs found on {url}")
            
            # Try to get additional details for each job
            for job in jobs:
                if job.get("job_url"):
                    try:
                        detailed_job = await self.scrape_job_details(job["job_url"])
                        if detailed_job and not detailed_job.get("error"):
                            job.update(detailed_job)
                        await asyncio.sleep(2)  # Respectful delay
                    except Exception as e:
                        logger.warning(f"Failed to get details for job {job.get('title', 'Unknown')}: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error scraping jobs from {org}: {str(e)}")
        
        return jobs
    
    def _parse_psu_job_listings(self, html_content: str, org: str) -> List[Dict[str, Any]]:
        """Parse PSU job listings from HTML content."""
        soup = self.parse_html(html_content)
        config = self.psu_configs[org]
        selectors = config["selectors"]
        
        jobs = []
        
        # Try multiple selector patterns for job cards
        job_cards = []
        for selector in selectors["job_cards"].split(", "):
            cards = self.extract_multiple(soup, selector.strip())
            if cards:
                job_cards = cards
                break
        
        # If no job cards found, try generic patterns
        if not job_cards:
            generic_selectors = [
                "table tr", ".table tr", "tbody tr",  # Table-based listings
                ".content p", ".main p",  # Paragraph-based listings
                "ul li", ".list li",  # List-based listings
                "div[class*='job']", "div[class*='career']", "div[class*='vacancy']"
            ]
            
            for selector in generic_selectors:
                cards = self.extract_multiple(soup, selector)
                if len(cards) > 3:  # Reasonable number of job entries
                    job_cards = cards
                    break
        
        logger.info(f"Found {len(job_cards)} job cards for {org}")
        
        for card in job_cards:
            try:
                job_data = self._extract_psu_job_data(card, selectors, org)
                if job_data and job_data.get("title"):
                    jobs.append(job_data)
            except Exception as e:
                logger.warning(f"Error parsing job card from {org}: {str(e)}")
                continue
        
        return jobs
    
    def _extract_psu_job_data(self, card_soup, selectors: Dict[str, str], org: str) -> Optional[Dict[str, Any]]:
        """Extract job data from a PSU job card/row."""
        try:
            # Handle table row format (common in government sites)
            if card_soup.name == "tr":
                return self._extract_from_table_row(card_soup, org)
            
            # Handle paragraph format
            if card_soup.name == "p":
                return self._extract_from_paragraph(card_soup, org)
            
            # Handle list item format
            if card_soup.name == "li":
                return self._extract_from_list_item(card_soup, org)
            
            # Handle div/card format
            return self._extract_from_card_format(card_soup, selectors, org)
            
        except Exception as e:
            logger.warning(f"Error extracting PSU job data: {str(e)}")
            return None
    
    def _extract_from_table_row(self, row_soup, org: str) -> Optional[Dict[str, Any]]:
        """Extract job data from table row format."""
        cells = row_soup.find_all(["td", "th"])
        if len(cells) < 3:  # Need at least 3 columns for meaningful data
            return None
        
        # Common table structures for government jobs
        # Structure 1: Post | Qualification | Experience | Deadline
        # Structure 2: S.No | Post | Location | Qualification | Last Date
        
        job_data = {
            "title": "",
            "qualification": "",
            "experience": "",
            "application_deadline": "",
            "location": "",
            "vacancy_count": "",
            "advertisement_number": "",
            "job_url": ""
        }
        
        # Extract text from all cells
        cell_texts = [self.clean_text(cell.get_text()) for cell in cells]
        
        # Try to identify columns based on content patterns
        for i, text in enumerate(cell_texts):
            if not text or text.lower() in ["s.no", "sr.no", "sl.no"]:
                continue
                
            # Check if this looks like a job title
            if not job_data["title"] and self._looks_like_job_title(text):
                job_data["title"] = text
                
            # Check if this looks like qualification
            elif not job_data["qualification"] and self._looks_like_qualification(text):
                job_data["qualification"] = text
                
            # Check if this looks like experience
            elif not job_data["experience"] and self._looks_like_experience(text):
                job_data["experience"] = text
                
            # Check if this looks like a date
            elif not job_data["application_deadline"] and self._looks_like_date(text):
                job_data["application_deadline"] = text
                
            # Check if this looks like location
            elif not job_data["location"] and self._looks_like_location(text):
                job_data["location"] = text
        
        # If title is still empty, use first non-empty cell
        if not job_data["title"] and cell_texts:
            job_data["title"] = cell_texts[0] if cell_texts[0] else "Government Position"
        
        # Look for job URL in row
        link = row_soup.find("a")
        if link and link.get("href"):
            job_data["job_url"] = self.build_absolute_url(link.get("href"))
        
        job_data.update({
            "organization": self.psu_configs[org]["name"],
            "source_organization": org,
            "scraped_at": datetime.now().isoformat(),
            "is_government_job": True,
            "is_psu_job": True
        })
        
        return job_data if job_data["title"] else None
    
    def _extract_from_paragraph(self, para_soup, org: str) -> Optional[Dict[str, Any]]:
        """Extract job data from paragraph format."""
        text = self.clean_text(para_soup.get_text())
        if len(text) < 20:  # Too short to be meaningful
            return None
        
        job_data = {
            "title": "",
            "description": text,
            "organization": self.psu_configs[org]["name"],
            "source_organization": org,
            "scraped_at": datetime.now().isoformat(),
            "is_government_job": True,
            "is_psu_job": True
        }
        
        # Try to extract title from beginning of text
        sentences = text.split(".")
        if sentences:
            potential_title = sentences[0].strip()
            if len(potential_title) < 100:  # Reasonable title length
                job_data["title"] = potential_title
        
        # Look for job URL in paragraph
        link = para_soup.find("a")
        if link and link.get("href"):
            job_data["job_url"] = self.build_absolute_url(link.get("href"))
        
        return job_data if job_data["title"] else None
    
    def _extract_from_list_item(self, li_soup, org: str) -> Optional[Dict[str, Any]]:
        """Extract job data from list item format."""
        text = self.clean_text(li_soup.get_text())
        if len(text) < 10:
            return None
        
        job_data = {
            "title": text,
            "organization": self.psu_configs[org]["name"],
            "source_organization": org,
            "scraped_at": datetime.now().isoformat(),
            "is_government_job": True,
            "is_psu_job": True
        }
        
        # Look for job URL in list item
        link = li_soup.find("a")
        if link and link.get("href"):
            job_data["job_url"] = self.build_absolute_url(link.get("href"))
        
        return job_data
    
    def _extract_from_card_format(self, card_soup, selectors: Dict[str, str], org: str) -> Optional[Dict[str, Any]]:
        """Extract job data from card/div format."""
        job_data = {
            "title": "",
            "department": "",
            "location": "",
            "qualification": "",
            "experience": "",
            "application_deadline": "",
            "vacancy_count": "",
            "advertisement_number": "",
            "job_url": ""
        }
        
        # Extract data using selectors
        for field, selector in selectors.items():
            if selector and field in job_data:
                value = self.extract_text(card_soup, selector)
                if value:
                    job_data[field] = value
        
        # Extract job URL
        if "details_url" in selectors:
            url = self.extract_attribute(card_soup, selectors["details_url"], "href")
            if url:
                job_data["job_url"] = self.build_absolute_url(url)
        
        job_data.update({
            "organization": self.psu_configs[org]["name"],
            "source_organization": org,
            "scraped_at": datetime.now().isoformat(),
            "is_government_job": True,
            "is_psu_job": True
        })
        
        return job_data if job_data["title"] else None
    
    def _looks_like_job_title(self, text: str) -> bool:
        """Check if text looks like a job title."""
        text_lower = text.lower()
        
        # Check for common job title patterns
        job_indicators = [
            "officer", "assistant", "manager", "engineer", "executive",
            "specialist", "analyst", "consultant", "supervisor", "operator",
            "technician", "clerk", "secretary", "coordinator", "advisor",
            "inspector", "trainee", "apprentice", "graduate", "post"
        ]
        
        return any(indicator in text_lower for indicator in job_indicators)
    
    def _looks_like_qualification(self, text: str) -> bool:
        """Check if text looks like educational qualification."""
        text_lower = text.lower()
        
        qualification_indicators = [
            "degree", "diploma", "graduation", "b.tech", "b.e", "m.tech",
            "mba", "mca", "bca", "bsc", "msc", "ba", "ma", "phd",
            "engineering", "graduate", "10th", "12th", "intermediate",
            "bachelor", "master", "doctorate", "professional"
        ]
        
        return any(indicator in text_lower for indicator in qualification_indicators)
    
    def _looks_like_experience(self, text: str) -> bool:
        """Check if text looks like experience requirement."""
        text_lower = text.lower()
        
        experience_patterns = [
            r'\d+\s*year', r'\d+\s*yr', r'experience', r'exp\.',
            r'fresher', r'minimum.*year', r'maximum.*year'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in experience_patterns)
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        # Common date patterns in Indian government websites
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or DD-MM-YYYY
            r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}'  # YYYY/MM/DD or YYYY-MM-DD
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in date_patterns)
    
    def _looks_like_location(self, text: str) -> bool:
        """Check if text looks like a location."""
        text_lower = text.lower()
        
        # Common Indian cities and location indicators
        location_indicators = [
            "delhi", "mumbai", "bangalore", "hyderabad", "chennai", "kolkata",
            "pune", "ahmedabad", "gurgaon", "noida", "kochi", "lucknow",
            "jaipur", "bhopal", "indore", "nagpur", "visakhapatnam",
            "all india", "pan india", "various locations", "multiple locations",
            "headquarters", "regional office", "branch office", "unit", "plant"
        ]
        
        return any(indicator in text_lower for indicator in location_indicators)
    
    async def scrape_job_details(self, job_url: str) -> Dict[str, Any]:
        """Scrape detailed job information from job URL."""
        if not job_url or not self.is_valid_url(job_url):
            return {"error": "Invalid job URL"}
        
        try:
            # Fetch job details page
            result = await self.fetch_with_browser(job_url, wait_for="body")
            
            if not result.success:
                return {"error": f"Failed to fetch job details: {result.error}"}
            
            # Parse job details
            job_details = self._parse_psu_job_details(result.data["html"])
            job_details["job_url"] = job_url
            job_details["scraped_at"] = datetime.now().isoformat()
            
            return job_details
            
        except Exception as e:
            logger.error(f"Error scraping job details from {job_url}: {str(e)}")
            return {"error": str(e)}
    
    def _parse_psu_job_details(self, html_content: str) -> Dict[str, Any]:
        """Parse detailed PSU job information from job details page."""
        soup = self.parse_html(html_content)
        
        details = {
            "title": "",
            "organization": "",
            "department": "",
            "location": "",
            "qualification": "",
            "experience": "",
            "age_limit": "",
            "vacancy_count": "",
            "application_deadline": "",
            "advertisement_number": "",
            "description": "",
            "requirements": "",
            "salary_range": "",
            "reservation_details": "",
            "application_fee": "",
            "selection_process": "",
            "how_to_apply": "",
            "important_dates": "",
            "is_ex_servicemen_eligible": False,
            "ex_servicemen_age_relaxation": ""
        }
        
        # Try to extract content from common patterns
        content_selectors = [
            ".content", ".main-content", ".job-details", ".notification-details",
            ".vacancy-details", ".recruitment-details", "#content", "#main"
        ]
        
        main_content = None
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                main_content = content
                break
        
        if not main_content:
            main_content = soup
        
        # Extract all text content
        full_text = main_content.get_text()
        
        # Extract title from page title or heading
        title_selectors = ["h1", "h2", ".title", ".job-title", ".post-title", "title"]
        for selector in title_selectors:
            title = self.extract_text(soup, selector)
            if title and len(title) < 200:
                details["title"] = title
                break
        
        # Extract structured information using patterns
        details.update(self._extract_structured_info(full_text))
        
        # Check for ex-servicemen eligibility
        details["is_ex_servicemen_eligible"] = self._check_ex_servicemen_eligibility(full_text)
        if details["is_ex_servicemen_eligible"]:
            details["ex_servicemen_age_relaxation"] = self._extract_age_relaxation_info(full_text)
        
        return details
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information from job text using patterns."""
        info = {}
        text_lower = text.lower()
        
        # Extract advertisement/notification number
        advt_patterns = [
            r'advertisement\s+no\.?\s*:?\s*([a-zA-Z0-9/\-]+)',
            r'notification\s+no\.?\s*:?\s*([a-zA-Z0-9/\-]+)',
            r'advt\.?\s+no\.?\s*:?\s*([a-zA-Z0-9/\-]+)',
            r'ref\.?\s+no\.?\s*:?\s*([a-zA-Z0-9/\-]+)'
        ]
        
        for pattern in advt_patterns:
            match = re.search(pattern, text_lower)
            if match:
                info["advertisement_number"] = match.group(1)
                break
        
        # Extract vacancy count
        vacancy_patterns = [
            r'total\s+posts?\s*:?\s*(\d+)',
            r'no\.?\s+of\s+posts?\s*:?\s*(\d+)',
            r'vacancies?\s*:?\s*(\d+)',
            r'(\d+)\s+posts?'
        ]
        
        for pattern in vacancy_patterns:
            match = re.search(pattern, text_lower)
            if match:
                info["vacancy_count"] = match.group(1)
                break
        
        # Extract application deadline
        deadline_patterns = [
            r'last\s+date\s*:?\s*([0-9]{1,2}[/-][0-9
