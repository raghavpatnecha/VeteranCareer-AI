import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import re
import json
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from app.models.user import User
from app.models.job import Job
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class JobMatchResult:
    """Data class for job matching results."""
    job_id: int
    job_title: str
    company_name: str
    location: str
    match_score: float
    skill_match_score: float
    experience_match_score: float
    location_match_score: float
    preference_match_score: float
    salary_match_score: float
    match_breakdown: Dict[str, float]
    match_reasons: List[str]
    improvement_suggestions: List[str]


@dataclass
class UserProfile:
    """Data class for user profile information used in matching."""
    user_id: int
    skills: List[str]
    experience_years: int
    preferred_locations: List[str]
    current_location: str
    preferred_industries: List[str]
    preferred_job_titles: List[str]
    expected_salary_min: Optional[int]
    expected_salary_max: Optional[int]
    willing_to_relocate: bool
    preferred_work_type: str
    service_branch: str
    rank: str
    military_skills: List[str]
    civilian_skills: List[str]
    technical_skills: List[str]


class JobMatcher:
    """Job matching algorithm using scikit-learn for scoring job-profile compatibility."""
    
    def __init__(self):
        self.skill_vectorizer = None
        self.location_geocoder = None
        self.location_cache = {}
        self.ml_model = None
        self.scaler = MinMaxScaler()
        
        # Weights for different matching components
        self.component_weights = {
            'skills': 0.35,
            'experience': 0.20,
            'location': 0.15,
            'preferences': 0.20,
            'salary': 0.10
        }
        
        # Initialize components
        self._initialize_vectorizer()
        self._initialize_geocoder()
        self._load_location_mappings()
        
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer for skill matching."""
        self.skill_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        
    def _initialize_geocoder(self):
        """Initialize geocoder for location matching."""
        try:
            self.location_geocoder = Nominatim(user_agent="veterancareer-matcher/1.0")
        except Exception as e:
            logger.warning(f"Failed to initialize geocoder: {str(e)}")
            self.location_geocoder = None
    
    def _load_location_mappings(self):
        """Load common location mappings and coordinates."""
        self.location_mappings = {
            # Major Indian cities with coordinates
            "delhi": {"lat": 28.6139, "lon": 77.2090, "aliases": ["new delhi", "ncr", "gurgaon", "noida"]},
            "mumbai": {"lat": 19.0760, "lon": 72.8777, "aliases": ["bombay", "navi mumbai", "thane"]},
            "bangalore": {"lat": 12.9716, "lon": 77.5946, "aliases": ["bengaluru", "whitefield", "electronic city"]},
            "chennai": {"lat": 13.0827, "lon": 80.2707, "aliases": ["madras"]},
            "hyderabad": {"lat": 17.3850, "lon": 78.4867, "aliases": ["secunderabad", "cyberabad"]},
            "pune": {"lat": 18.5204, "lon": 73.8567, "aliases": ["poona"]},
            "kolkata": {"lat": 22.5726, "lon": 88.3639, "aliases": ["calcutta"]},
            "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "aliases": ["amdavad"]},
            "jaipur": {"lat": 26.9124, "lon": 75.7873, "aliases": []},
            "lucknow": {"lat": 26.8467, "lon": 80.9462, "aliases": []},
            "kanpur": {"lat": 26.4499, "lon": 80.3319, "aliases": []},
            "nagpur": {"lat": 21.1458, "lon": 79.0882, "aliases": []},
            "indore": {"lat": 22.7196, "lon": 75.8577, "aliases": []},
            "thane": {"lat": 19.2183, "lon": 72.9781, "aliases": []},
            "bhopal": {"lat": 23.2599, "lon": 77.4126, "aliases": []},
            "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "aliases": ["vizag"]},
            "pimpri": {"lat": 18.6298, "lon": 73.7997, "aliases": ["pimpri chinchwad"]},
            "patna": {"lat": 25.5941, "lon": 85.1376, "aliases": []},
            "vadodara": {"lat": 22.3072, "lon": 73.1812, "aliases": ["baroda"]},
            "ludhiana": {"lat": 30.9010, "lon": 75.8573, "aliases": []},
            "agra": {"lat": 27.1767, "lon": 78.0081, "aliases": []},
            "nashik": {"lat": 19.9975, "lon": 73.7898, "aliases": []},
            "faridabad": {"lat": 28.4089, "lon": 77.3178, "aliases": []},
            "meerut": {"lat": 28.9845, "lon": 77.7064, "aliases": []},
            "rajkot": {"lat": 22.3039, "lon": 70.8022, "aliases": []},
            "kalyan": {"lat": 19.2437, "lon": 73.1355, "aliases": ["kalyan dombivli"]},
            "vasai": {"lat": 19.4912, "lon": 72.8054, "aliases": ["vasai virar"]},
            "varanasi": {"lat": 25.3176, "lon": 82.9739, "aliases": ["benares"]},
            "srinagar": {"lat": 34.0837, "lon": 74.7973, "aliases": []},
            "dhanbad": {"lat": 23.7957, "lon": 86.4304, "aliases": []},
            "jodhpur": {"lat": 26.2389, "lon": 73.0243, "aliases": []},
            "amritsar": {"lat": 31.6340, "lon": 74.8723, "aliases": []},
            "raipur": {"lat": 21.2514, "lon": 81.6296, "aliases": []},
            "allahabad": {"lat": 25.4358, "lon": 81.8463, "aliases": ["prayagraj"]},
            "coimbatore": {"lat": 11.0168, "lon": 76.9558, "aliases": []},
            "jabalpur": {"lat": 23.1815, "lon": 79.9864, "aliases": []},
            "gwalior": {"lat": 26.2183, "lon": 78.1828, "aliases": []},
            "vijayawada": {"lat": 16.5062, "lon": 80.6480, "aliases": []},
            "madurai": {"lat": 9.9252, "lon": 78.1198, "aliases": []},
            "guwahati": {"lat": 26.1445, "lon": 91.7362, "aliases": []},
            "chandigarh": {"lat": 30.7333, "lon": 76.7794, "aliases": []},
            "hubli": {"lat": 15.3647, "lon": 75.1240, "aliases": ["hubli dharwad"]},
            "mysore": {"lat": 12.2958, "lon": 76.6394, "aliases": ["mysuru"]},
            "tiruchirappalli": {"lat": 10.7905, "lon": 78.7047, "aliases": ["trichy"]},
            "bareilly": {"lat": 28.3670, "lon": 79.4304, "aliases": []},
            "aligarh": {"lat": 27.8974, "lon": 78.0880, "aliases": []},
            "salem": {"lat": 11.6643, "lon": 78.1460, "aliases": []},
            "mira bhayandar": {"lat": 19.2952, "lon": 72.8544, "aliases": []},
            "thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "aliases": ["trivandrum"]},
            "bhiwandi": {"lat": 19.3002, "lon": 73.0682, "aliases": []},
            "saharanpur": {"lat": 29.9680, "lon": 77.5552, "aliases": []},
            "gorakhpur": {"lat": 26.7606, "lon": 83.3732, "aliases": []},
            "guntur": {"lat": 16.3067, "lon": 80.4365, "aliases": []},
            "bikaner": {"lat": 28.0229, "lon": 73.3119, "aliases": []},
            "amravati": {"lat": 20.9374, "lon": 77.7796, "aliases": []},
            "noida": {"lat": 28.5355, "lon": 77.3910, "aliases": []},
            "jamshedpur": {"lat": 22.8046, "lon": 86.2029, "aliases": []},
            "bhilai": {"lat": 21.1938, "lon": 81.3509, "aliases": []},
            "cuttack": {"lat": 20.4625, "lon": 85.8828, "aliases": []},
            "firozabad": {"lat": 27.1592, "lon": 78.3957, "aliases": []},
            "kochi": {"lat": 9.9312, "lon": 76.2673, "aliases": ["cochin"]},
            "nellore": {"lat": 14.4426, "lon": 79.9865, "aliases": []},
            "bhavnagar": {"lat": 21.7645, "lon": 72.1519, "aliases": []},
            "dehradun": {"lat": 30.3165, "lon": 78.0322, "aliases": []},
            "durgapur": {"lat": 23.5204, "lon": 87.3119, "aliases": []},
            "asansol": {"lat": 23.6839, "lon": 86.9753, "aliases": []},
            "rourkela": {"lat": 22.2604, "lon": 84.8536, "aliases": []},
            "nanded": {"lat": 19.1383, "lon": 77.3210, "aliases": []},
            "kolhapur": {"lat": 16.7050, "lon": 74.2433, "aliases": []},
            "ajmer": {"lat": 26.4499, "lon": 74.6399, "aliases": []},
            "akola": {"lat": 20.7002, "lon": 77.0082, "aliases": []},
            "gulbarga": {"lat": 17.3297, "lon": 76.8343, "aliases": ["kalaburagi"]},
            "jamnagar": {"lat": 22.4707, "lon": 70.0577, "aliases": []},
            "ujjain": {"lat": 23.1765, "lon": 75.7885, "aliases": []},
            "loni": {"lat": 28.7333, "lon": 77.2833, "aliases": []},
            "siliguri": {"lat": 26.7271, "lon": 88.3953, "aliases": []},
            "jhansi": {"lat": 25.4484, "lon": 78.5685, "aliases": []},
            "ulhasnagar": {"lat": 19.2215, "lon": 73.1645, "aliases": []},
            "jammu": {"lat": 32.7266, "lon": 74.8570, "aliases": []},
            "sangli": {"lat": 16.8524, "lon": 74.5815, "aliases": ["sangli miraj kupwad"]},
            "mangalore": {"lat": 12.9141, "lon": 74.8560, "aliases": ["mangaluru"]},
            "erode": {"lat": 11.3410, "lon": 77.7172, "aliases": []},
            "belgaum": {"lat": 15.8497, "lon": 74.4977, "aliases": ["belagavi"]},
            "ambattur": {"lat": 13.1143, "lon": 80.1548, "aliases": []},
            "tirunelveli": {"lat": 8.7139, "lon": 77.7567, "aliases": []},
            "malegaon": {"lat": 20.5579, "lon": 74.5287, "aliases": []},
            "gaya": {"lat": 24.7955, "lon": 85.0002, "aliases": []},
            "jalgaon": {"lat": 21.0077, "lon": 75.5626, "aliases": []},
            "udaipur": {"lat": 24.5854, "lon": 73.7125, "aliases": []},
            "maheshtala": {"lat": 22.5049, "lon": 88.2482, "aliases": []}
        }
    
    def calculate_job_match_score(self, user_profile: UserProfile, job: Job) -> JobMatchResult:
        """
        Calculate comprehensive job match score between user profile and job.
        
        Args:
            user_profile: User profile information
            job: Job information
            
        Returns:
            JobMatchResult with detailed scoring breakdown
        """
        try:
            # Calculate individual component scores
            skill_score = self._calculate_skill_match(user_profile, job)
            experience_score = self._calculate_experience_match(user_profile, job)
            location_score = self._calculate_location_match(user_profile, job)
            preference_score = self._calculate_preference_match(user_profile, job)
            salary_score = self._calculate_salary_match(user_profile, job)
            
            # Calculate weighted overall score
            overall_score = (
                skill_score * self.component_weights['skills'] +
                experience_score * self.component_weights['experience'] +
                location_score * self.component_weights['location'] +
                preference_score * self.component_weights['preferences'] +
                salary_score * self.component_weights['salary']
            )
            
            # Generate match breakdown and reasons
            match_breakdown = {
                'skills': skill_score,
                'experience': experience_score,
                'location': location_score,
                'preferences': preference_score,
                'salary': salary_score,
                'overall': overall_score
            }
            
            match_reasons = self._generate_match_reasons(match_breakdown, user_profile, job)
            improvement_suggestions = self._generate_improvement_suggestions(match_breakdown, user_profile, job)
            
            return JobMatchResult(
                job_id=job.id,
                job_title=job.title,
                company_name=job.company_name,
                location=job.location,
                match_score=round(overall_score, 2),
                skill_match_score=round(skill_score, 2),
                experience_match_score=round(experience_score, 2),
                location_match_score=round(location_score, 2),
                preference_match_score=round(preference_score, 2),
                salary_match_score=round(salary_score, 2),
                match_breakdown=match_breakdown,
                match_reasons=match_reasons,
                improvement_suggestions=improvement_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error calculating job match score: {str(e)}")
            # Return minimal match result
            return JobMatchResult(
                job_id=job.id,
                job_title=job.title,
                company_name=job.company_name,
                location=job.location,
                match_score=0.0,
                skill_match_score=0.0,
                experience_match_score=0.0,
                location_match_score=0.0,
                preference_match_score=0.0,
                salary_match_score=0.0,
                match_breakdown={},
                match_reasons=["Error in calculation"],
                improvement_suggestions=[]
            )
    
    def _calculate_skill_match(self, user_profile: UserProfile, job: Job) -> float:
        """Calculate skill match score between user and job."""
        try:
            # Combine all user skills
            user_skills = []
            user_skills.extend(user_profile.skills or [])
            user_skills.extend(user_profile.military_skills or [])
            user_skills.extend(user_profile.civilian_skills or [])
            user_skills.extend(user_profile.technical_skills or [])
            
            # Get job required skills
            job_skills = job.required_skills or []
            
            if not user_skills or not job_skills:
                return 30.0  # Base score for no skill data
            
            # Create skill texts for vectorization
            user_skill_text = " ".join([skill.lower() for skill in user_skills])
            job_skill_text = " ".join([skill.lower() for skill in job_skills])
            
            # Add job description and title for broader skill matching
            job_text_full = f"{job_skill_text} {job.title.lower()} {job.description.lower() if job.description else ''}"
            
            # Calculate cosine similarity
            documents = [user_skill_text, job_text_full]
            
            try:
                tfidf_matrix = self.skill_vectorizer.fit_transform(documents)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                similarity_score = similarity_matrix[0][1]
            except Exception:
                # Fallback to simple keyword matching
                similarity_score = self._simple_skill_match(user_skills, job_skills)
            
            # Convert to 0-100 scale with adjustments
            base_score = similarity_score * 100
            
            # Bonus for exact skill matches
            exact_matches = len(set([s.lower() for s in user_skills]) & set([s.lower() for s in job_skills]))
            if exact_matches > 0:
                base_score += min(exact_matches * 5, 20)  # Up to 20 bonus points
            
            # Bonus for veteran-specific jobs
            if job.veteran_preference:
                base_score += 10
            
            # Bonus for government/PSU jobs (good for ex-servicemen)
            if job.government_job or job.psu_job:
                base_score += 5
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating skill match: {str(e)}")
            return 30.0
    
    def _simple_skill_match(self, user_skills: List[str], job_skills: List[str]) -> float:
        """Simple skill matching fallback method."""
        if not user_skills or not job_skills:
            return 0.3
        
        user_skills_lower = set(skill.lower() for skill in user_skills)
        job_skills_lower = set(skill.lower() for skill in job_skills)
        
        # Calculate Jaccard similarity
        intersection = len(user_skills_lower & job_skills_lower)
        union = len(user_skills_lower | job_skills_lower)
        
        return intersection / union if union > 0 else 0.3
    
    def _calculate_experience_match(self, user_profile: UserProfile, job: Job) -> float:
        """Calculate experience match score."""
        try:
            user_experience = user_profile.experience_years or 0
            
            # Get job experience requirements
            min_exp = job.min_experience_years or 0
            max_exp = job.max_experience_years or 50
            
            # Perfect match if within range
            if min_exp <= user_experience <= max_exp:
                return 100.0
            
            # Calculate score based on deviation
            if user_experience < min_exp:
                # Under-qualified
                deficit = min_exp - user_experience
                if deficit <= 2:
                    return 80.0  # Close enough
                elif deficit <= 5:
                    return 60.0
                else:
                    return 30.0
            else:
                # Over-qualified
                excess = user_experience - max_exp
                if excess <= 3:
                    return 90.0  # Slightly overqualified is good
                elif excess <= 7:
                    return 75.0
                else:
                    return 50.0  # Significantly overqualified
                    
        except Exception as e:
            logger.error(f"Error calculating experience match: {str(e)}")
            return 50.0
    
    def _calculate_location_match(self, user_profile: UserProfile, job: Job) -> float:
        """Calculate location match score."""
        try:
            job_location = job.location.lower() if job.location else ""
            user_current = user_profile.current_location.lower() if user_profile.current_location else ""
            user_preferred = [loc.lower() for loc in (user_profile.preferred_locations or [])]
            
            # Remote jobs get high score
            if job.is_remote:
                return 95.0
            
            # Exact match with current location
            if user_current and self._locations_match(user_current, job_location):
                return 100.0
            
            # Match with preferred locations
            for preferred_loc in user_preferred:
                if self._locations_match(preferred_loc, job_location):
                    return 95.0
            
            # Calculate distance-based score if willing to relocate
            if user_profile.willing_to_relocate:
                distance_score = self._calculate_distance_score(user_current, job_location)
                if distance_score > 0:
                    return min(85.0, distance_score)
            
            # Check for city aliases and nearby locations
            alias_score = self._check_location_aliases(user_current, job_location, user_preferred)
            if alias_score > 0:
                return alias_score
            
            # Default score based on willingness to relocate
            return 60.0 if user_profile.willing_to_relocate else 20.0
            
        except Exception as e:
            logger.error(f"Error calculating location match: {str(e)}")
            return 50.0
    
    def _locations_match(self, loc1: str, loc2: str) -> bool:
        """Check if two location strings match."""
        if not loc1 or not loc2:
            return False
        
        # Clean location strings
        loc1_clean = re.sub(r'[^\w\s]', ' ', loc1.lower()).strip()
        loc2_clean = re.sub(r'[^\w\s]', ' ', loc2.lower()).strip()
        
        # Direct match
        if loc1_clean == loc2_clean:
            return True
        
        # Check if one contains the other
        if loc1_clean in loc2_clean or loc2_clean in loc1_clean:
            return True
        
        # Check for common words (at least 2 matching words for cities)
        words1 = set(loc1_clean.split())
        words2 = set(loc2_clean.split())
        common_words = words1 & words2
        
        return len(common_words) >= 2
    
    def _calculate_distance_score(self, user_location: str, job_location: str) -> float:
        """Calculate location score based on distance."""
        try:
            user_coords = self._get_coordinates(user_location)
            job_coords = self._get_coordinates(job_location)
            
            if user_coords and job_coords:
                distance = geodesic(user_coords, job_coords).kilometers
                
                # Score based on distance (0-100 scale)
                if distance <= 50:
                    return 90.0
                elif distance <= 100:
                    return 80.0
                elif distance <= 200:
                    return 70.0
                elif distance <= 500:
                    return 60.0
                elif distance <= 1000:
                    return 50.0
                else:
                    return 30.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating distance score: {str(e)}")
            return 0.0
    
    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location."""
        if not location:
            return None
        
        location_lower = location.lower().strip()
        
        # Check cache first
        if location_lower in self.location_cache:
            return self.location_cache[location_lower]
        
        # Check predefined mappings
        for city, data in self.location_mappings.items():
            if city in location_lower or any(alias in location_lower for alias in data["aliases"]):
                coords = (data["lat"], data["lon"])
                self.location_cache[location_lower] = coords
                return coords
        
        # Use geocoder as fallback (with rate limiting)
        if self.location_geocoder:
            try:
                location_obj = self.location_geocoder.geocode(location, timeout=5)
                if location_obj:
                    coords = (location_obj.latitude, location_obj.longitude)
                    self.location_cache[location_lower] = coords
                    return coords
            except Exception as e:
                logger.warning(f"Geocoding failed for {location}: {str(e)}")
        
        return None
    
    def _check_location_aliases(self, user_location: str, job_location: str, preferred_locations: List[str]) -> float:
        """Check for location aliases and nearby areas."""
        all_user_locations = [user_location] + preferred_locations
        
        for user_loc in all_user_locations:
            if not user_loc:
                continue
                
            for city, data in self.location_mappings.items():
                # Check if user location matches a major city
                if city in user_loc.lower() or any(alias in user_loc.lower() for alias in data["aliases"]):
                    # Check if job location is in the same metro area
                    if city in job_location.lower() or any(alias in job_location.lower() for alias in data["aliases"]):
                        return 85.0
        
        return 0.0
    
    def _calculate_preference_match(self, user_profile: UserProfile, job: Job) -> float:
        """Calculate preference match score."""
        try:
            total_score = 0.0
            factors_count = 0
            
            # Industry preference match
            if user_profile.preferred_industries and job.industry:
                industry_match = any(
                    industry.lower() in job.industry.lower() 
                    for industry in user_profile.preferred_industries
                )
                total_score += 100.0 if industry_match else 30.0
                factors_count += 1
            
            # Job title preference match
            if user_profile.preferred_job_titles and job.title:
                title_match = any(
                    title.lower() in job.title.lower() 
                    for title in user_profile.preferred_job_titles
                )
                total_score += 100.0 if title_match else 40.0
                factors_count += 1
            
            # Work type preference match
            if user_profile.preferred_work_type and job.job_type:
                work_type_match = user_profile.preferred_work_type.lower() == job.job_type.lower()
                total_score += 100.0 if work_type_match else 50.0
                factors_count += 1
            
            # Company type preferences (for ex-servicemen)
            company_score = 60.0  # Base score
            if job.government_job or job.psu_job:
                company_score = 90.0  # Higher score for govt/PSU jobs
            elif job.veteran_preference:
                company_score = 85.0  # High score for veteran-friendly companies
            
            total_score += company_score
            factors_count += 1
            
            # Job level matching based on military rank/experience
            if user_profile.rank and job.experience_level:
                level_score = self._calculate_level_match(user_profile.rank, job.experience_level)
                total_score += level_score
                factors_count += 1
            
            return total_score / max(factors_count, 1)
            
        except Exception as e:
            logger.error(f"Error calculating preference match: {str(e)}")
            return 50.0
    
    def _calculate_level_match(self, military_rank: str, job_level: str) -> float:
        """Calculate job level match based on military rank."""
        rank_lower = military_rank.lower()
        level_lower = job_level.lower()
        
        # Officer ranks typically match with senior/executive positions
        officer_ranks = ['captain', 'major', 'colonel', 'general', 'commander', 'lieutenant colonel']
        if any(rank in rank_lower for rank in officer_ranks):
            if 'senior' in level_lower or 'executive' in level_lower or 'manager' in level_lower:
                return 90.0
            elif 'mid' in level_lower:
                return 70.0
            else:
                return 50.0
        
        # NCO ranks typically match with mid-level positions
        nco_ranks = ['sergeant', 'corporal', 'petty officer']
        if any(rank in rank_lower for rank in nco_ranks):
            if 'mid' in level_lower or 'senior' in level_lower:
                return 85.0
            elif 'entry' in level_lower:
                return 60.0
            else:
                return 70.0
        
        # Default matching
        return 60.0
    
    def _calculate_salary_match(self, user_profile: UserProfile, job: Job) -> float:
        """Calculate salary match score."""
        try:
            user_min = user_profile.expected_salary_min
            user_max = user_profile.expected_salary_max
            job_min = job.salary_min
            job_max = job.salary_max
            
            # If no salary information available
            if not any([user_min, user_max, job_min, job_max]):
                return 70.0  # Neutral score
            
            # If job salary not disclosed
            if not job_min and not job_max:
                return 60.0
            
            # If user has no salary expectations
            if not user_min and not user_max:
                return 75.0  # Good score when user is flexible
            
            # Calculate overlap and scoring
            job_min_val = job_min or 0
            job_max_val = job_max or float('inf')
            user_min_val = user_min or 0
            user_max_val = user_max or float('inf')
            
            # Check for salary range overlap
            overlap_start = max(user_min_val, job_min_val)
            overlap_end = min(user_max_val, job_max_val)
            
            if overlap_start <= overlap_end:
                # There is overlap - calculate quality of match
                if job_min and job_max and user_min and user_max:
                    # Both ranges fully specified
                    job_range = job_max - job_min
                    user_range = user_max - user_min
                    overlap_size = overlap_end - overlap_start
                    
                    # Score based on overlap percentage
                    if job_range > 0:
                        overlap_ratio = overlap_size / job_range
                        if overlap_ratio >= 0.8:
                            return 95.0
                        elif overlap_ratio >= 0.5:
                            return 85.0
                        else:
                            return 75.0
                    else:
                        return 90.0  # Exact salary match
                else:
                    return 80.0  # Partial overlap with incomplete data
            else:
                # No overlap - check how far apart they are
                if user_max_val < job_min_val:
                    # User expectation too low
                    gap = (job_min_val - user_max_val) / max(user_max_val, 1)
                    if gap <= 0.2:
                        return 60.0  # Close enough
                    elif gap <= 0.5:
                        return 40.0
                    else:
                        return 20.0
                else:
                    # User expectation too high
                    gap = (user_min_val - job_max_val) / max(job_max_val, 1)
                    if gap <= 0.3:
                        return 50.0  # Slightly overexpectation
                    elif gap <= 0.7:
                        return 30.0
                    else:
                        return 10.0
            
        except Exception as e:
            logger.error(f"Error calculating salary match: {str(e)}")
            return 50.0

    def _generate_match_reasons(self, match_breakdown: Dict[str, float], user_profile: UserProfile, job: Job) -> List[str]:
        """Generate reasons for job match based on scoring breakdown."""
        reasons = []
        
        try:
            # Skill match reasons
            if match_breakdown.get('skills', 0) >= 80:
                reasons.append("Strong skill alignment with job requirements")
            elif match_breakdown.get('skills', 0) >= 60:
                reasons.append("Good skill match with some gaps")
            
            # Experience match reasons
            if match_breakdown.get('experience', 0) >= 90:
                reasons.append("Perfect experience level match")
            elif match_breakdown.get('experience', 0) >= 70:
                reasons.append("Suitable experience level")
            
            # Location match reasons
            if match_breakdown.get('location', 0) >= 95:
                if job.is_remote:
                    reasons.append("Remote work opportunity")
                else:
                    reasons.append("Excellent location match")
            elif match_breakdown.get('location', 0) >= 80:
                reasons.append("Good location compatibility")
            
            # Preference match reasons
            if match_breakdown.get('preferences', 0) >= 85:
                if job.government_job or job.psu_job:
                    reasons.append("Government/PSU role suitable for ex-servicemen")
                if job.veteran_preference:
                    reasons.append("Veteran-friendly employer")
            
            # Salary match reasons
            if match_breakdown.get('salary', 0) >= 80:
                reasons.append("Salary expectations well aligned")
            elif match_breakdown.get('salary', 0) >= 60:
                reasons.append("Reasonable salary compatibility")
            
            # Military background advantages
            if user_profile.service_branch and (job.veteran_preference or job.government_job):
                reasons.append(f"Military background ({user_profile.service_branch}) valued by employer")
            
            # Leadership experience
            if user_profile.rank and any(keyword in user_profile.rank.lower() for keyword in ['officer', 'captain', 'major', 'colonel']):
                reasons.append("Leadership experience from military service")
            
            return reasons[:5]  # Limit to top 5 reasons
            
        except Exception as e:
            logger.error(f"Error generating match reasons: {str(e)}")
            return ["Good overall compatibility"]

    def _generate_improvement_suggestions(self, match_breakdown: Dict[str, float], user_profile: UserProfile, job: Job) -> List[str]:
        """Generate suggestions for improving job match."""
        suggestions = []
        
        try:
            # Skill improvement suggestions
            if match_breakdown.get('skills', 0) < 60:
                job_skills = job.required_skills or []
                user_skills = (user_profile.skills or []) + (user_profile.technical_skills or [])
                missing_skills = set(skill.lower() for skill in job_skills) - set(skill.lower() for skill in user_skills)
                if missing_skills:
                    suggestions.append(f"Consider developing skills in: {', '.join(list(missing_skills)[:3])}")
                else:
                    suggestions.append("Highlight relevant military skills that transfer to civilian roles")
            
            # Experience suggestions
            if match_breakdown.get('experience', 0) < 50:
                if user_profile.experience_years < (job.min_experience_years or 0):
                    suggestions.append("Emphasize military experience and leadership roles to compensate for civilian experience gap")
                else:
                    suggestions.append("Consider roles that better match your experience level")
            
            # Location suggestions
            if match_breakdown.get('location', 0) < 60:
                if not user_profile.willing_to_relocate:
                    suggestions.append("Consider expanding your location preferences or remote work options")
                else:
                    suggestions.append("Look for similar roles in your preferred locations")
            
            # Salary expectations
            if match_breakdown.get('salary', 0) < 50:
                if user_profile.expected_salary_min and job.salary_max:
                    if user_profile.expected_salary_min > job.salary_max:
                        suggestions.append("Consider adjusting salary expectations for this role level")
                suggestions.append("Research market rates for similar positions in your area")
            
            # Career transition advice
            if match_breakdown.get('overall', 0) < 70:
                suggestions.append("Consider transitional roles that bridge military and civilian experience")
                suggestions.append("Highlight soft skills like leadership, discipline, and teamwork from military service")
            
            # Profile enhancement
            if not user_profile.civilian_skills:
                suggestions.append("Add civilian equivalent skills to better match job requirements")
            
            return suggestions[:4]  # Limit to top 4 suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}")
            return ["Continue developing relevant skills for your target roles"]
