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
class JobListing:
    """Data class for job listing information."""
    title: str
    company: str
    location: str
    experience: str
    salary: Optional[str] = None
    job_type: Optional[str] = None
    posted_date: Optional[str] = None
    job_url: str = ""
    description: str = ""
    requirements: str = ""
    skills: List[str] = None
    is_veteran_friendly: bool = False
    source_portal: str = ""
    external_id: str = ""

    def __post_init__(self):
        if self.skills is None:
            self.skills = []


class JobPortalScraper(BaseScraper):
    """Scraper for popular Indian job portals like Naukri, Indeed, Monster, etc."""
    
    def __init__(self, portal_name: str = "multi_portal"):
        # Configure rate limiting for job portals
        rate_config = RateLimitConfig(
            requests_per_minute=20,
            requests_per_hour=300,
            delay_between_requests=3.0,
            random_delay_range=(2.0, 5.0)
        )
        
        super().__init__(
            name=portal_name,
            base_url="https://www.naukri.com",  # Default to Naukri
            rate_limit_config=rate_config
        )
        
        # Portal configurations
        self.portal_configs = {
            "naukri": {
                "base_url": "https://www.naukri.com",
                "search_url": "https://www.naukri.com/jobs-in-{location}",
                "search_params": {"k": "{keywords}", "l": "{location}"},
                "selectors": {
                    "job_cards": ".srp-jobtuple-wrapper",
                    "title": ".title a",
                    "company": ".comp-name",
                    "location": ".locationsContainer",
                    "experience": ".exp",
                    "salary": ".salary",
                    "posted_date": ".job-post-day",
                    "job_url": ".title a"
                }
            },
            "indeed": {
                "base_url": "https://in.indeed.com",
                "search_url": "https://in.indeed.com/jobs",
                "search_params": {"q": "{keywords}", "l": "{location}"},
                "selectors": {
                    "job_cards": "[data-jk]",
                    "title": "[data-testid='job-title']",
                    "company": "[data-testid='company-name']",
                    "location": "[data-testid='job-location']",
                    "salary": "[data-testid='salary-snippet']",
                    "job_url": "[data-testid='job-title'] a"
                }
            },
            "monster": {
                "base_url": "https://www.monsterindia.com",
                "search_url": "https://www.monsterindia.com/search/{keywords}",
                "selectors": {
                    "job_cards": ".job-result",
                    "title": ".job-tittle a",
                    "company": ".company-name",
                    "location": ".job-location",
                    "experience": ".experience",
                    "salary": ".salary",
                    "job_url": ".job-tittle a"
                }
            },
            "foundit": {
                "base_url": "https://www.foundit.in",
                "search_url": "https://www.foundit.in/jobs/{keywords}",
                "selectors": {
                    "job_cards": ".srpResultCardContainer",
                    "title": ".jobTitle a",
                    "company": ".companyName",
                    "location": ".jobLocation",
                    "experience": ".expSalary",
                    "job_url": ".jobTitle a"
                }
            }
        }
        
        # Keywords that indicate veteran/military preference
        self.veteran_keywords = [
            "veteran", "ex-servicemen", "military", "army", "navy", "air force",
            "defense", "defence", "security clearance", "government", "psu",
            "public sector", "paramilitary", "armed forces", "commissioned",
            "non-commissioned", "officer", "jawans", "forces background",
            "military experience", "service personnel", "retired personnel"
        ]
        
        # Industry keywords relevant to veterans
        self.veteran_industries = [
            "defense", "security", "aviation", "logistics", "project management",
            "operations", "maintenance", "engineering", "telecommunications",
            "it services", "manufacturing", "infrastructure", "consulting",
            "training", "education", "healthcare", "administration"
        ]
        
    async def scrape_jobs(
        self,
        keywords: str = "ex-servicemen",
        location: str = "india",
        portals: List[str] = None,
        max_pages: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scrape jobs from multiple job portals.
        
        Args:
            keywords: Search keywords for jobs
            location: Location to search in
            portals: List of portals to scrape (default: all)
            max_pages: Maximum pages to scrape per portal
            
        Returns:
            List of job dictionaries
        """
        if portals is None:
            portals = ["naukri", "indeed", "monster", "foundit"]
        
        all_jobs = []
        
        for portal in portals:
            if portal not in self.portal_configs:
                logger.warning(f"Unknown portal: {portal}")
                continue
                
            try:
                logger.info(f"Scraping jobs from {portal}")
                portal_jobs = await self._scrape_portal_jobs(
                    portal, keywords, location, max_pages
                )
                
                # Add source portal to each job
                for job in portal_jobs:
                    job["source_portal"] = portal
                
                all_jobs.extend(portal_jobs)
                logger.info(f"Found {len(portal_jobs)} jobs on {portal}")
                
            except Exception as e:
                logger.error(f"Error scraping {portal}: {str(e)}")
                continue
        
        # Filter for veteran-friendly positions
        veteran_jobs = self._filter_veteran_friendly_jobs(all_jobs)
        
        logger.info(f"Total jobs found: {len(all_jobs)}, Veteran-friendly: {len(veteran_jobs)}")
        
        return veteran_jobs
    
    async def _scrape_portal_jobs(
        self,
        portal: str,
        keywords: str,
        location: str,
        max_pages: int
    ) -> List[Dict[str, Any]]:
        """Scrape jobs from a specific portal."""
        config = self.portal_configs[portal]
        jobs = []
        
        # Update base URL for the portal
        self.base_url = config["base_url"]
        
        for page in range(1, max_pages + 1):
            try:
                search_url = self._build_search_url(portal, keywords, location, page)
                
                # Fetch search results page
                result = await self.fetch_with_browser(
                    search_url,
                    wait_for=config["selectors"]["job_cards"]
                )
                
                if not result.success:
                    logger.warning(f"Failed to fetch page {page} from {portal}: {result.error}")
                    break
                
                # Parse job listings from the page
                page_jobs = self._parse_job_listings(result.data["html"], portal)
                
                if not page_jobs:
                    logger.info(f"No more jobs found on page {page} of {portal}")
                    break
                
                jobs.extend(page_jobs)
                logger.info(f"Found {len(page_jobs)} jobs on page {page} of {portal}")
                
                # Add delay between pages
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping page {page} from {portal}: {str(e)}")
                break
        
        return jobs
    
    def _build_search_url(self, portal: str, keywords: str, location: str, page: int = 1) -> str:
        """Build search URL for a specific portal."""
        config = self.portal_configs[portal]
        
        if portal == "naukri":
            # Naukri-specific URL building
            base_url = config["search_url"].format(location=location.replace(" ", "-").lower())
            params = {
                "k": keywords,
                "experience": "0,50",  # All experience levels
                "qp": f"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",  # All experience ranges
                "sPage": str(page)
            }
            
        elif portal == "indeed":
            base_url = config["search_url"]
            params = {
                "q": keywords,
                "l": location,
                "start": str((page - 1) * 10)  # Indeed uses start parameter
            }
            
        elif portal == "monster":
            base_url = config["search_url"].format(keywords=keywords.replace(" ", "-"))
            params = {
                "location": location,
                "page": str(page)
            }
            
        elif portal == "foundit":
            base_url = config["search_url"].format(keywords=keywords.replace(" ", "-"))
            params = {
                "locations": location,
                "page": str(page)
            }
        
        # Build URL with parameters
        if params:
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            return f"{base_url}?{param_string}"
        
        return base_url
    
    def _parse_job_listings(self, html_content: str, portal: str) -> List[Dict[str, Any]]:
        """Parse job listings from HTML content."""
        soup = self.parse_html(html_content)
        config = self.portal_configs[portal]
        selectors = config["selectors"]
        
        jobs = []
        job_cards = self.extract_multiple(soup, selectors["job_cards"])
        
        for card in job_cards:
            try:
                job_data = self._extract_job_data(card, selectors, portal)
                if job_data:
                    jobs.append(job_data)
            except Exception as e:
                logger.warning(f"Error parsing job card from {portal}: {str(e)}")
                continue
        
        return jobs
    
    def _extract_job_data(self, card_soup, selectors: Dict[str, str], portal: str) -> Optional[Dict[str, Any]]:
        """Extract job data from a job card."""
        try:
            # Extract basic job information
            title = self.extract_text(card_soup, selectors["title"])
            company = self.extract_text(card_soup, selectors["company"])
            location = self.extract_text(card_soup, selectors["location"])
            
            # Skip if essential fields are missing
            if not title or not company:
                return None
            
            # Extract optional fields
            experience = self.extract_text(card_soup, selectors.get("experience", ""), "Not specified")
            salary = self.extract_text(card_soup, selectors.get("salary", ""), "Not disclosed")
            posted_date = self.extract_text(card_soup, selectors.get("posted_date", ""), "")
            
            # Extract job URL
            job_url = ""
            if "job_url" in selectors:
                relative_url = self.extract_attribute(card_soup, selectors["job_url"], "href")
                if relative_url:
                    job_url = self.build_absolute_url(relative_url)
            
            # Create job data dictionary
            job_data = {
                "title": self.clean_text(title),
                "company": self.clean_text(company),
                "location": self.clean_text(location),
                "experience": self.clean_text(experience),
                "salary": self.clean_text(salary),
                "posted_date": self.clean_text(posted_date),
                "job_url": job_url,
                "source_portal": portal,
                "scraped_at": datetime.now().isoformat(),
                "external_id": self._generate_external_id(job_url, title, company)
            }
            
            return job_data
            
        except Exception as e:
            logger.warning(f"Error extracting job data: {str(e)}")
            return None
    
    def _generate_external_id(self, job_url: str, title: str, company: str) -> str:
        """Generate external ID for job."""
        if job_url:
            # Try to extract job ID from URL
            parsed_url = urlparse(job_url)
            
            # For Naukri URLs
            if "naukri.com" in parsed_url.netloc:
                path_parts = parsed_url.path.split("/")
                for part in path_parts:
                    if part.startswith("job-listings-"):
                        return part
            
            # For Indeed URLs
            elif "indeed.com" in parsed_url.netloc:
                query_params = parse_qs(parsed_url.query)
                if "jk" in query_params:
                    return query_params["jk"][0]
        
        # Fallback: create hash from title and company
        import hashlib
        hash_input = f"{title}_{company}".lower().replace(" ", "_")
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def scrape_job_details(self, job_url: str) -> Dict[str, Any]:
        """Scrape detailed job information from job URL."""
        if not job_url or not self.is_valid_url(job_url):
            return {"error": "Invalid job URL"}
        
        try:
            # Determine portal from URL
            portal = self._identify_portal_from_url(job_url)
            
            # Fetch job details page
            result = await self.fetch_with_browser(job_url)
            
            if not result.success:
                return {"error": f"Failed to fetch job details: {result.error}"}
            
            # Parse job details based on portal
            job_details = self._parse_job_details(result.data["html"], portal)
            job_details["job_url"] = job_url
            job_details["scraped_at"] = datetime.now().isoformat()
            
            return job_details
            
        except Exception as e:
            logger.error(f"Error scraping job details from {job_url}: {str(e)}")
            return {"error": str(e)}
    
    def _identify_portal_from_url(self, url: str) -> str:
        """Identify job portal from URL."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if "naukri.com" in domain:
            return "naukri"
        elif "indeed.com" in domain:
            return "indeed"
        elif "monsterindia.com" in domain:
            return "monster"
        elif "foundit.in" in domain:
            return "foundit"
        else:
            return "unknown"
    
    def _parse_job_details(self, html_content: str, portal: str) -> Dict[str, Any]:
        """Parse detailed job information from job details page."""
        soup = self.parse_html(html_content)
        
        # Portal-specific parsing
        if portal == "naukri":
            return self._parse_naukri_job_details(soup)
        elif portal == "indeed":
            return self._parse_indeed_job_details(soup)
        elif portal == "monster":
            return self._parse_monster_job_details(soup)
        elif portal == "foundit":
            return self._parse_foundit_job_details(soup)
        else:
            return self._parse_generic_job_details(soup)
    
    def _parse_naukri_job_details(self, soup) -> Dict[str, Any]:
        """Parse job details from Naukri job page."""
        details = {
            "title": self.extract_text(soup, ".jd-header-title", ""),
            "company": self.extract_text(soup, ".jd-header-comp-name", ""),
            "location": self.extract_text(soup, ".location-details", ""),
            "experience": self.extract_text(soup, ".exp-details", ""),
            "salary": self.extract_text(soup, ".salary-details", ""),
            "description": self.extract_text(soup, ".dang-inner-html", ""),
            "job_type": self.extract_text(soup, ".job-type", ""),
            "posted_date": self.extract_text(soup, ".job-post-date", ""),
            "skills": [],
            "requirements": ""
        }
        
        # Extract skills
        skill_elements = self.extract_multiple(soup, ".chip-clickable")
        details["skills"] = [self.clean_text(skill.get_text()) for skill in skill_elements]
        
        return details
    
    def _parse_indeed_job_details(self, soup) -> Dict[str, Any]:
        """Parse job details from Indeed job page."""
        details = {
            "title": self.extract_text(soup, "[data-testid='job-title']", ""),
            "company": self.extract_text(soup, "[data-testid='company-name']", ""),
            "location": self.extract_text(soup, "[data-testid='job-location']", ""),
            "salary": self.extract_text(soup, "[data-testid='salary-snippet']", ""),
            "description": self.extract_text(soup, "#jobDescriptionText", ""),
            "job_type": "",
            "posted_date": "",
            "skills": [],
            "requirements": ""
        }
        
        return details
    
    def _parse_monster_job_details(self, soup) -> Dict[str, Any]:
        """Parse job details from Monster job page."""
        details = {
            "title": self.extract_text(soup, ".job-title", ""),
            "company": self.extract_text(soup, ".company-name", ""),
            "location": self.extract_text(soup, ".location", ""),
            "experience": self.extract_text(soup, ".experience-range", ""),
            "salary": self.extract_text(soup, ".salary-range", ""),
            "description": self.extract_text(soup, ".job-description", ""),
            "skills": [],
            "requirements": ""
        }
        
        return details
    
    def _parse_foundit_job_details(self, soup) -> Dict[str, Any]:
        """Parse job details from Foundit job page."""
        details = {
            "title": self.extract_text(soup, ".job-header-title", ""),
            "company": self.extract_text(soup, ".company-name", ""),
            "location": self.extract_text(soup, ".job-location", ""),
            "experience": self.extract_text(soup, ".experience", ""),
            "salary": self.extract_text(soup, ".salary", ""),
            "description": self.extract_text(soup, ".job-description", ""),
            "skills": [],
            "requirements": ""
        }
        
        return details
    
    def _parse_generic_job_details(self, soup) -> Dict[str, Any]:
        """Generic job details parsing for unknown portals."""
        # Try common selectors
        common_selectors = {
            "title": ["h1", ".job-title", ".title", "[data-testid='job-title']"],
            "company": [".company", ".company-name", "[data-testid='company-name']"],
            "location": [".location", ".job-location", "[data-testid='job-location']"],
            "description": [".description", ".job-description", "#jobDescriptionText"]
        }
        
        details = {}
        
        for field, selectors in common_selectors.items():
            for selector in selectors:
                text = self.extract_text(soup, selector)
                if text:
                    details[field] = text
                    break
            
            if field not in details:
                details[field] = ""
        
        details.update({
            "skills": [],
            "requirements": "",
            "salary": "",
            "experience": "",
            "job_type": "",
            "posted_date": ""
        })
        
        return details
    
    def _filter_veteran_friendly_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter jobs that are suitable for veterans."""
        veteran_jobs = []
        
        for job in jobs:
            is_veteran_friendly = self._is_veteran_friendly_job(job)
            job["is_veteran_friendly"] = is_veteran_friendly
            job["veteran_match_score"] = self._calculate_veteran_match_score(job)
            
            # Include job if it has veteran relevance or high match score
            if is_veteran_friendly or job["veteran_match_score"] >= 50:
                veteran_jobs.append(job)
        
        # Sort by veteran match score (highest first)
        veteran_jobs.sort(key=lambda x: x["veteran_match_score"], reverse=True)
        
        return veteran_jobs
    
    def _is_veteran_friendly_job(self, job: Dict[str, Any]) -> bool:
        """Check if job is explicitly veteran-friendly."""
        # Check title and description for veteran keywords
        text_to_check = f"{job.get('title', '')} {job.get('description', '')}".lower()
        
        for keyword in self.veteran_keywords:
            if keyword.lower() in text_to_check:
                return True
        
        # Check company names for government/PSU organizations
        company = job.get("company", "").lower()
        government_indicators = [
            "government", "psu", "public sector", "ministry", "department",
            "hal", "bel", "drdo", "isro", "ongc", "ntpc", "bhel", "gail",
            "indian railways", "air india", "bpcl", "hpcl", "coal india"
        ]
        
        for indicator in government_indicators:
            if indicator in company:
                return True
        
        return False
    
    def _calculate_veteran_match_score(self, job: Dict[str, Any]) -> float:
        """Calculate veteran match score for a job (0-100)."""
        score = 0
        
        # Check for explicit veteran preference (high score)
        if self._is_veteran_friendly_job(job):
            score += 40
        
        # Check job title for relevant roles
        title = job.get("title", "").lower()
        relevant_titles = [
            "manager", "officer", "supervisor", "coordinator", "analyst",
            "administrator", "consultant", "specialist", "engineer",
            "security", "operations", "logistics", "maintenance",
            "project", "team lead", "instructor", "trainer"
        ]
        
        for title_keyword in relevant_titles:
            if title_keyword in title:
                score += 10
                break
        
        # Check for relevant industries/companies
        company = job.get("company", "").lower()
        location = job.get("location", "").lower()
        
        # Bonus for government/defense/PSU companies
        if any(keyword in company for keyword in ["government", "psu", "defence", "defense", "security"]):
            score += 20
        
        # Bonus for metro cities (more opportunities)
        metro_cities = ["delhi", "mumbai", "bangalore", "chennai", "hyderabad", "pune", "kolkata"]
        if any(city in location for city in metro_cities):
            score += 10
        
        # Check experience requirements
        experience = job.get("experience", "").lower()
        
        # Bonus for jobs accepting varied experience levels
        if any(phrase in experience for phrase in ["0-", "fresher", "any", "all levels"]):
            score += 10
        
        # Ensure score is within 0-100 range
        return min(100, max(0, score))
    
    async def get_trending_skills(self, portal: str = "naukri") -> List[Dict[str, Any]]:
        """Get trending skills for veterans from job postings."""
        try:
            # Scrape recent job postings for skill analysis
            recent_jobs = await self.scrape_jobs(
                keywords="ex-servicemen OR military OR veteran",
                max_pages=3
            )
            
            # Analyze skills from job descriptions
            skill_frequency = {}
            
            for job in recent_jobs:
                skills = job.get("skills", [])
                description = job.get("description", "")
                
                # Extract skills from description using common patterns
                extracted_skills = self._extract_skills_from_text(description)
                all_skills = skills + extracted_skills
                
                for skill in all_skills:
                    skill_clean = skill.strip().lower()
                    if skill_clean and len(skill_clean) > 2:
                        skill_frequency[skill_clean] = skill_frequency.get(skill_clean, 0) + 1
            
            # Sort skills by frequency and return top ones
            trending_skills = [
                {"skill": skill, "frequency": freq, "jobs_count": freq}
                for skill, freq in sorted(skill_frequency.items(), 
                                        key=lambda x: x[1], reverse=True)[:20]
            ]
            
            return trending_skills
            
        except Exception as e:
            logger.error(f"Error getting trending skills: {str(e)}")
            return []
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from job description text."""
        if not text:
            return []
        
        # Common skill patterns and keywords
        skill_patterns = [
            r'\b(?:MS|Microsoft)\s+(?:Office|Excel|Word|PowerPoint|Outlook)\b',
            r'\b(?:Java|Python|JavaScript|C\+\+|SQL|HTML|CSS)\b',
            r'\b(?:Project Management|Leadership|Communication|Teamwork)\b',
            r'\b(?:SAP|Oracle|Salesforce|Tableau|Power BI)\b',
            r'\b(?:AWS|Azure|Google Cloud|Docker|Kubernetes)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        # Common skills for veterans
        veteran_skills = [
            "leadership", "project management", "team management", "operations",
            "logistics", "maintenance", "security", "training", "coordination",
            "administration", "planning", "execution", "communication",
            "problem solving", "decision making", "risk management"
        ]
        
        for skill in veteran_skills:
            if skill in text_lower:
                skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    async def search_jobs_by_skills(
        self,
        skills: List[str],
        location: str = "india",
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Search jobs by specific skills relevant to veterans."""
        # Convert skills to search keywords
        skill_keywords = " OR ".join(skills)
        search_query = f"({skill_keywords}) AND (ex-servicemen OR military OR government OR PSU)"
        
        jobs = await self.scrape_jobs(
            keywords=search_query,
            location=location,
            max_pages=3
        )
        
        # Score jobs based on skill match
        for job in jobs:
            job["skill_match_score"] = self._calculate_skill_match_score(job, skills)
        
        # Sort by skill match score and limit results
        jobs.sort(key=lambda x: x.get("skill_match_score", 0), reverse=True)
        
        return jobs[:max_results]
    
    def _calculate_skill_match_score(self, job: Dict[str, Any], target_skills: List[str]) -> float:
        """Calculate how well job matches target skills."""
        job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        
        matches = 0
        for skill in target_skills:
            if skill.lower() in job_text:
                matches += 1
        
        return (matches / len(target_skills)) * 100 if target_skills else 0
