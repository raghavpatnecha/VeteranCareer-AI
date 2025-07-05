import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from statistics import median, mean
from collections import defaultdict, Counter

from app.models.user import User
from app.models.job import Job
from app.models.skill_mapping import SkillMapping
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CareerAdvice:
    """Data class for career guidance recommendations."""
    recommended_industries: List[Dict[str, Any]]
    career_paths: List[Dict[str, Any]]
    skill_gaps: List[Dict[str, str]]
    salary_insights: Dict[str, Any]
    next_steps: List[str]
    timeline_recommendations: List[Dict[str, Any]]
    training_suggestions: List[Dict[str, str]]
    networking_advice: List[str]
    market_trends: Dict[str, Any]
    confidence_score: float


@dataclass
class SalaryBenchmark:
    """Data class for salary benchmarking information."""
    role_title: str
    industry: str
    location: str
    experience_level: str
    salary_min: int
    salary_max: int
    salary_median: int
    market_percentiles: Dict[str, int]
    growth_projection: float
    demand_level: str
    source_data_count: int


@dataclass
class CareerTrajectory:
    """Data class for career progression planning."""
    current_role: str
    target_role: str
    progression_path: List[Dict[str, Any]]
    estimated_timeline: Dict[str, int]
    required_skills: List[str]
    recommended_certifications: List[str]
    salary_progression: List[Dict[str, Any]]
    industry_mobility: Dict[str, float]
    success_factors: List[str]
    potential_challenges: List[str]


class CareerAdvisor:
    """Career guidance module with industry-specific advice and trajectory planning."""
    
    def __init__(self):
        self.salary_model = None
        self.career_path_model = None
        self.scaler = StandardScaler()
        self.industry_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        
        # Industry-specific data
        self.industry_data = self._load_industry_data()
        self.salary_benchmarks = {}
        self.career_paths = self._load_career_paths()
        self.skill_demand_trends = {}
        
        # Ex-servicemen specific mappings
        self.military_civilian_roles = self._load_military_civilian_mappings()
        self.veteran_friendly_industries = self._load_veteran_industries()
        
        # Market trend data
        self.market_trends = self._load_market_trends()
        
    def _load_industry_data(self) -> Dict[str, Any]:
        """Load comprehensive industry data and characteristics."""
        return {
            "technology": {
                "growth_rate": 0.15,
                "avg_salary_range": (600000, 2500000),
                "skill_demand": ["python", "java", "cloud computing", "ai/ml", "devops"],
                "veteran_suitability": 0.85,
                "remote_work_percentage": 0.70,
                "job_security": 0.80,
                "career_progression": "fast",
                "key_employers": ["TCS", "Infosys", "Google", "Microsoft", "Amazon"],
                "entry_barriers": "medium",
                "certification_value": "high"
            },
            "cybersecurity": {
                "growth_rate": 0.22,
                "avg_salary_range": (800000, 3000000),
                "skill_demand": ["network security", "ethical hacking", "compliance", "risk assessment"],
                "veteran_suitability": 0.95,
                "remote_work_percentage": 0.60,
                "job_security": 0.95,
                "career_progression": "fast",
                "key_employers": ["IBM", "Deloitte", "PwC", "Wipro", "Accenture"],
                "entry_barriers": "high",
                "certification_value": "very high"
            },
            "defense_aerospace": {
                "growth_rate": 0.08,
                "avg_salary_range": (700000, 2200000),
                "skill_demand": ["systems engineering", "project management", "quality assurance", "compliance"],
                "veteran_suitability": 0.98,
                "remote_work_percentage": 0.30,
                "job_security": 0.90,
                "career_progression": "steady",
                "key_employers": ["HAL", "BEL", "DRDO", "L&T Defence", "Tata Advanced Systems"],
                "entry_barriers": "low",
                "certification_value": "medium"
            },
            "logistics_supply_chain": {
                "growth_rate": 0.12,
                "avg_salary_range": (500000, 1800000),
                "skill_demand": ["supply chain management", "logistics optimization", "inventory management", "analytics"],
                "veteran_suitability": 0.90,
                "remote_work_percentage": 0.40,
                "job_security": 0.75,
                "career_progression": "steady",
                "key_employers": ["Blue Dart", "Delhivery", "Amazon Logistics", "Flipkart", "Mahindra Logistics"],
                "entry_barriers": "low",
                "certification_value": "medium"
            },
            "project_management": {
                "growth_rate": 0.10,
                "avg_salary_range": (800000, 2500000),
                "skill_demand": ["project management", "agile", "risk management", "stakeholder management"],
                "veteran_suitability": 0.92,
                "remote_work_percentage": 0.50,
                "job_security": 0.80,
                "career_progression": "steady",
                "key_employers": ["Accenture", "Deloitte", "PwC", "KPMG", "EY"],
                "entry_barriers": "medium",
                "certification_value": "high"
            },
            "aviation": {
                "growth_rate": 0.06,
                "avg_salary_range": (600000, 2000000),
                "skill_demand": ["flight operations", "aviation maintenance", "safety management", "regulatory compliance"],
                "veteran_suitability": 0.93,
                "remote_work_percentage": 0.20,
                "job_security": 0.70,
                "career_progression": "slow",
                "key_employers": ["Air India", "IndiGo", "SpiceJet", "GoAir", "Vistara"],
                "entry_barriers": "high",
                "certification_value": "very high"
            },
            "manufacturing": {
                "growth_rate": 0.07,
                "avg_salary_range": (450000, 1500000),
                "skill_demand": ["operations management", "quality control", "lean manufacturing", "automation"],
                "veteran_suitability": 0.88,
                "remote_work_percentage": 0.25,
                "job_security": 0.75,
                "career_progression": "steady",
                "key_employers": ["Tata Motors", "Mahindra", "Bajaj Auto", "Hero MotoCorp", "Maruti Suzuki"],
                "entry_barriers": "low",
                "certification_value": "medium"
            },
            "consulting": {
                "growth_rate": 0.13,
                "avg_salary_range": (900000, 3500000),
                "skill_demand": ["strategic thinking", "problem solving", "client management", "domain expertise"],
                "veteran_suitability": 0.85,
                "remote_work_percentage": 0.60,
                "job_security": 0.70,
                "career_progression": "fast",
                "key_employers": ["McKinsey", "BCG", "Bain", "Deloitte", "PwC"],
                "entry_barriers": "high",
                "certification_value": "medium"
            },
            "telecommunications": {
                "growth_rate": 0.09,
                "avg_salary_range": (550000, 1800000),
                "skill_demand": ["network engineering", "telecommunications", "5G technology", "network security"],
                "veteran_suitability": 0.87,
                "remote_work_percentage": 0.45,
                "job_security": 0.75,
                "career_progression": "steady",
                "key_employers": ["Airtel", "Jio", "Vodafone Idea", "BSNL", "Nokia"],
                "entry_barriers": "medium",
                "certification_value": "high"
            },
            "government_public_sector": {
                "growth_rate": 0.05,
                "avg_salary_range": (400000, 1200000),
                "skill_demand": ["administration", "policy analysis", "public management", "compliance"],
                "veteran_suitability": 0.95,
                "remote_work_percentage": 0.20,
                "job_security": 0.95,
                "career_progression": "slow",
                "key_employers": ["Central Government", "State Governments", "PSUs", "Municipal Corporations"],
                "entry_barriers": "medium",
                "certification_value": "low"
            }
        }
    
    def _load_career_paths(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load common career progression paths."""
        return {
            "technology": [
                {
                    "level": "entry",
                    "roles": ["Software Developer", "Junior Analyst", "Technical Support"],
                    "experience": "0-2 years",
                    "salary_range": (400000, 800000),
                    "key_skills": ["programming", "problem solving", "basic tools"]
                },
                {
                    "level": "mid",
                    "roles": ["Senior Developer", "Team Lead", "Systems Analyst"],
                    "experience": "3-7 years",
                    "salary_range": (800000, 1500000),
                    "key_skills": ["advanced programming", "team leadership", "architecture"]
                },
                {
                    "level": "senior",
                    "roles": ["Tech Lead", "Solution Architect", "Engineering Manager"],
                    "experience": "8-12 years",
                    "salary_range": (1500000, 2500000),
                    "key_skills": ["strategic thinking", "people management", "technical leadership"]
                },
                {
                    "level": "executive",
                    "roles": ["Director Engineering", "VP Technology", "CTO"],
                    "experience": "12+ years",
                    "salary_range": (2500000, 5000000),
                    "key_skills": ["business strategy", "organizational leadership", "innovation"]
                }
            ],
            "project_management": [
                {
                    "level": "entry",
                    "roles": ["Project Coordinator", "Assistant Project Manager", "PMO Analyst"],
                    "experience": "0-2 years",
                    "salary_range": (450000, 700000),
                    "key_skills": ["project coordination", "documentation", "basic PM tools"]
                },
                {
                    "level": "mid",
                    "roles": ["Project Manager", "Senior PM", "Program Coordinator"],
                    "experience": "3-7 years",
                    "salary_range": (700000, 1300000),
                    "key_skills": ["project management", "stakeholder management", "risk management"]
                },
                {
                    "level": "senior",
                    "roles": ["Senior Project Manager", "Program Manager", "PMO Head"],
                    "experience": "8-12 years",
                    "salary_range": (1300000, 2200000),
                    "key_skills": ["program management", "strategic planning", "team leadership"]
                },
                {
                    "level": "executive",
                    "roles": ["Director PMO", "VP Operations", "Chief Operating Officer"],
                    "experience": "12+ years",
                    "salary_range": (2200000, 4000000),
                    "key_skills": ["organizational strategy", "change management", "executive leadership"]
                }
            ],
            "cybersecurity": [
                {
                    "level": "entry",
                    "roles": ["Security Analyst", "Junior Penetration Tester", "SOC Analyst"],
                    "experience": "0-2 years",
                    "salary_range": (500000, 900000),
                    "key_skills": ["network security", "incident response", "security tools"]
                },
                {
                    "level": "mid",
                    "roles": ["Senior Security Analyst", "Security Engineer", "Compliance Manager"],
                    "experience": "3-7 years",
                    "salary_range": (900000, 1600000),
                    "key_skills": ["advanced security", "compliance", "risk assessment"]
                },
                {
                    "level": "senior",
                    "roles": ["Security Architect", "Security Manager", "CISO Deputy"],
                    "experience": "8-12 years",
                    "salary_range": (1600000, 2800000),
                    "key_skills": ["security architecture", "team management", "strategic security"]
                },
                {
                    "level": "executive",
                    "roles": ["CISO", "VP Security", "Security Director"],
                    "experience": "12+ years",
                    "salary_range": (2800000, 5500000),
                    "key_skills": ["security strategy", "board communication", "organizational risk"]
                }
            ]
        }
    
    def _load_military_civilian_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load military role to civilian career mappings."""
        return {
            "infantry_officer": {
                "primary_civilian_roles": ["Operations Manager", "Project Manager", "Team Lead"],
                "industries": ["manufacturing", "logistics", "project_management"],
                "transferable_skills": ["leadership", "decision making", "crisis management"],
                "salary_adjustment": 0.9
            },
            "signals_officer": {
                "primary_civilian_roles": ["IT Manager", "Network Engineer", "Telecommunications Specialist"],
                "industries": ["technology", "telecommunications", "cybersecurity"],
                "transferable_skills": ["technical leadership", "systems thinking", "problem solving"],
                "salary_adjustment": 1.0
            },
            "logistics_officer": {
                "primary_civilian_roles": ["Supply Chain Manager", "Operations Manager", "Logistics Director"],
                "industries": ["logistics_supply_chain", "manufacturing", "consulting"],
                "transferable_skills": ["logistics planning", "resource optimization", "coordination"],
                "salary_adjustment": 1.1
            },
            "pilot": {
                "primary_civilian_roles": ["Commercial Pilot", "Aviation Manager", "Flight Operations Manager"],
                "industries": ["aviation", "logistics_supply_chain"],
                "transferable_skills": ["precision", "safety management", "decision making under pressure"],
                "salary_adjustment": 1.2
            },
            "engineer_officer": {
                "primary_civilian_roles": ["Engineering Manager", "Project Engineer", "Technical Consultant"],
                "industries": ["manufacturing", "defense_aerospace", "consulting"],
                "transferable_skills": ["technical expertise", "project management", "quality assurance"],
                "salary_adjustment": 1.0
            },
            "intelligence_officer": {
                "primary_civilian_roles": ["Data Analyst", "Security Analyst", "Risk Manager"],
                "industries": ["cybersecurity", "consulting", "technology"],
                "transferable_skills": ["analytical thinking", "risk assessment", "strategic analysis"],
                "salary_adjustment": 1.1
            },
            "medical_officer": {
                "primary_civilian_roles": ["Medical Director", "Healthcare Manager", "Clinical Consultant"],
                "industries": ["healthcare", "consulting"],
                "transferable_skills": ["medical expertise", "crisis management", "leadership"],
                "salary_adjustment": 1.0
            }
        }
    
    def _load_veteran_industries(self) -> List[Dict[str, Any]]:
        """Load industries particularly suitable for veterans."""
        return [
            {
                "industry": "cybersecurity",
                "suitability_score": 0.95,
                "reasons": ["Security clearance background", "Discipline", "Risk assessment skills"],
                "growth_potential": "very high",
                "hiring_preference": "strong"
            },
            {
                "industry": "defense_aerospace",
                "suitability_score": 0.98,
                "reasons": ["Domain expertise", "Security awareness", "Technical knowledge"],
                "growth_potential": "steady",
                "hiring_preference": "very strong"
            },
            {
                "industry": "project_management",
                "suitability_score": 0.92,
                "reasons": ["Leadership experience", "Planning skills", "Execution discipline"],
                "growth_potential": "high",
                "hiring_preference": "strong"
            },
            {
                "industry": "logistics_supply_chain",
                "suitability_score": 0.90,
                "reasons": ["Operations experience", "Resource management", "Logistics expertise"],
                "growth_potential": "high",
                "hiring_preference": "strong"
            },
            {
                "industry": "government_public_sector",
                "suitability_score": 0.95,
                "reasons": ["Government experience", "Policy understanding", "Public service orientation"],
                "growth_potential": "steady",
                "hiring_preference": "very strong"
            }
        ]
    
    def _load_market_trends(self) -> Dict[str, Any]:
        """Load current market trends and projections."""
        return {
            "hot_skills": {
                "technology": ["AI/ML", "Cloud Computing", "DevOps", "Data Science", "Cybersecurity"],
                "business": ["Digital Transformation", "Agile", "Change Management", "Data Analytics"],
                "emerging": ["Blockchain", "IoT", "Automation", "Sustainability", "Remote Leadership"]
            },
            "declining_skills": ["Legacy Systems", "Manual Processes", "Traditional Marketing"],
            "salary_trends": {
                "technology": {"trend": "increasing", "rate": 0.12},
                "cybersecurity": {"trend": "increasing", "rate": 0.18},
                "project_management": {"trend": "stable", "rate": 0.06},
                "manufacturing": {"trend": "stable", "rate": 0.04}
            },
            "remote_work_trends": {
                "2023": 0.40,
                "2024_projected": 0.45,
                "2025_projected": 0.50
            },
            "hiring_challenges": [
                "Skill shortages in tech",
                "Competition for experienced talent",
                "Rapid technology changes",
                "Remote work adaptation"
            ]
        }
    
    def generate_career_advice(self, user_profile: Dict[str, Any], preferences: Dict[str, Any] = None) -> CareerAdvice:
        """
        Generate comprehensive career advice for a user.
        
        Args:
            user_profile: User's background, skills, and experience
            preferences: User's career preferences and goals
            
        Returns:
            CareerAdvice object with personalized recommendations
        """
        try:
            preferences = preferences or {}
            
            # Analyze user's background and skills
            skill_analysis = self._analyze_user_skills(user_profile)
            
            # Get industry recommendations
            industry_recommendations = self._recommend_industries(user_profile, skill_analysis)
            
            # Generate career paths
            career_paths = self._generate_career_paths(user_profile, industry_recommendations)
            
            # Identify skill gaps
            skill_gaps = self._identify_skill_gaps(user_profile, industry_recommendations)
            
            # Get salary insights
            salary_insights = self._get_salary_insights(user_profile, industry_recommendations)
            
            # Generate next steps
            next_steps = self._generate_next_steps(user_profile, skill_gaps, career_paths)
            
            # Create timeline recommendations
            timeline_recommendations = self._create_timeline_recommendations(career_paths, skill_gaps)
            
            # Get training suggestions
            training_suggestions = self._get_training_suggestions(skill_gaps, industry_recommendations)
            
            # Generate networking advice
            networking_advice = self._generate_networking_advice(user_profile, industry_recommendations)
            
            # Get market trends
            relevant_market_trends = self._get_relevant_market_trends(industry_recommendations)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(user_profile, skill_analysis, industry_recommendations)
            
            return CareerAdvice(
                recommended_industries=industry_recommendations,
                career_paths=career_paths,
                skill_gaps=skill_gaps,
                salary_insights=salary_insights,
                next_steps=next_steps,
                timeline_recommendations=timeline_recommendations,
                training_suggestions=training_suggestions,
                networking_advice=networking_advice,
                market_trends=relevant_market_trends,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating career advice: {str(e)}")
            return self._generate_basic_advice(user_profile)
    
    def _analyze_user_skills(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's skills and categorize them."""
        analysis = {
            "technical_skills": [],
            "leadership_skills": [],
            "domain_skills": [],
            "soft_skills": [],
            "skill_strength": {},
            "skill_categories": {}
        }
        
        all_skills = []
        all_skills.extend(user_profile.get("military_skills", []))
        all_skills.extend(user_profile.get("civilian_skills", []))
        all_skills.extend(user_profile.get("technical_skills", []))
        
        # Categorize skills
        technical_keywords = ["programming", "software", "network", "database", "cloud", "security", "automation"]
        leadership_keywords = ["leadership", "management", "team", "project", "coordination", "supervision"]
        domain_keywords = ["logistics", "operations", "maintenance", "training", "planning", "analysis"]
        
        for skill in all_skills:
            skill_lower = skill.lower()
            
            if any(keyword in skill_lower for keyword in technical_keywords):
                analysis["technical_skills"].append(skill)
                analysis["skill_categories"][skill] = "technical"
            elif any(keyword in skill_lower for keyword in leadership_keywords):
                analysis["leadership_skills"].append(skill)
                analysis["skill_categories"][skill] = "leadership"
            elif any(keyword in skill_lower for keyword in domain_keywords):
                analysis["domain_skills"].append(skill)
                analysis["skill_categories"][skill] = "domain"
            else:
                analysis["soft_skills"].append(skill)
                analysis["skill_categories"][skill] = "soft"
            
            # Estimate skill strength based on experience and military background
            analysis["skill_strength"][skill] = self._estimate_skill_strength(skill, user_profile)
        
        return analysis
    
    def _estimate_skill_strength(self, skill: str, user_profile: Dict[str, Any]) -> float:
        """Estimate skill strength based on user's background."""
        base_strength = 0.6  # Base strength for listed skills
        
        # Military experience adds credibility
        years_of_service = user_profile.get("years_of_service", 0)
        if years_of_service > 0:
            base_strength += min(years_of_service * 0.05, 0.3)
        
        # Officer rank adds leadership skill strength
        rank = user_profile.get("rank", "").lower()
        if "officer" in rank or "commander" in rank:
            if "leadership" in skill.lower() or "management" in skill.lower():
                base_strength += 0.2
        
        # Technical roles add technical skill strength
        service_branch = user_profile.get("service_branch", "").lower()
        if "engineer" in service_branch or "signals" in service_branch or "electronics" in service_branch:
            if any(keyword in skill.lower() for keyword in ["technical", "engineering", "systems", "electronics"]):
                base_strength += 0.15
        
        return min(1.0, base_strength)
    
    def _recommend_industries(self, user_profile: Dict[str, Any], skill_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend suitable industries for the user."""
        recommendations = []
        
        # Get military background mapping
        military_role = self._identify_military_role(user_profile)
        military_mapping = self.military_civilian_roles.get(military_role, {})
        
        # Score each industry
        for industry, data in self.industry_data.items():
            score = self._calculate_industry_score(user_profile, skill_analysis, industry, data)
            
            if score > 0.5:  # Only recommend industries with decent fit
                recommendation = {
                    "industry": industry,
                    "score": round(score, 2),
                    "growth_rate": data["growth_rate"],
                    "salary_range": data["avg_salary_range"],
                    "veteran_suitability": data["veteran_suitability"],
                    "job_security": data["job_security"],
                    "remote_work_potential": data["remote_work_percentage"],
                    "key_employers": data["key_employers"][:3],
                    "entry_barriers": data["entry_barriers"],
                    "reasons": self._get_industry_fit_reasons(skill_analysis, industry, data),
                    "challenges": self._get_industry_challenges(user_profile, industry, data)
                }
                recommendations.append(recommendation)
        
        # Sort by score and return top 5
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]
    
    def _identify_military_role(self, user_profile: Dict[str, Any]) -> str:
        """Identify primary military role from user profile."""
        service_branch = user_profile.get("service_branch", "").lower()
        rank = user_profile.get("rank", "").lower()
        skills = [skill.lower() for skill in user_profile.get("military_skills", [])]
        
        # Logic to map to military role categories
        if "pilot" in service_branch or any("pilot" in skill for skill in skills):
            return "pilot"
        elif "signals" in service_branch or any("communication" in skill for skill in skills):
            return "signals_officer"
        elif "logistics" in service_branch or any("logistics" in skill for skill in skills):
            return "logistics_officer"
        elif "engineer" in service_branch or any("engineering" in skill for skill in skills):
            return "engineer_officer"
        elif "intelligence" in service_branch or any("intelligence" in skill for skill in skills):
            return "intelligence_officer"
        elif "medical" in service_branch or any("medical" in skill for skill in skills):
            return "medical_officer"
        else:
            return "infantry_officer"  # Default
    
    def _calculate_industry_score(self, user_profile: Dict[str, Any], skill_analysis: Dict[str, Any], industry: str, industry_data: Dict[str, Any]) -> float:
        """Calculate fit score for an industry."""
        score = 0.0
        
        # Base veteran suitability score
        score += industry_data["veteran_suitability"] * 0.3
        
        # Skill match score
        skill_match = self._calculate_skill_industry_match(skill_analysis, industry_data)
        score += skill_match * 0.4
        
        # Experience relevance
        experience_match = self._calculate_experience_relevance(user_profile, industry)
        score += experience_match * 0.2
        
        # Market attractiveness (growth, salary, security)
        market_score = (
            industry_data["growth_rate"] * 2 +  # Normalized growth rate
            (industry_data["avg_salary_range"][1] / 3000000) +  # Normalized max salary
            industry_data["job_security"]
        ) / 3
        score += market_score * 0.1
        
        return min(1.0, score)
    
    def _calculate_skill_industry_match(self, skill_analysis: Dict[str, Any], industry_data: Dict[str, Any]) -> float:
        """Calculate how well user's skills match industry requirements."""
        user_skills = set(skill.lower() for skill in 
                         skill_analysis["technical_skills"] + 
                         skill_analysis["leadership_skills"] + 
                         skill_analysis["domain_skills"])
        
        industry_skills = set(skill.lower() for skill in industry_data["skill_demand"])
        
        if not user_skills or not industry_skills:
            return 0.3  # Base score
        
        # Calculate overlap
        overlap = len(user_skills & industry_skills)
        total_required = len(industry_skills)
        
        match_score = overlap / total_required if total_required > 0 else 0.3
        
        # Bonus for leadership skills in management-heavy industries
        if skill_analysis["leadership_skills"] and industry_data.get("leadership_importance", 0.5) > 0.7:
            match_score += 0.2
        
        return min(1.0, match_score)
    
    def _calculate_experience_relevance(self, user_profile: Dict[str, Any], industry: str) -> float:
        """Calculate relevance of user's experience to the industry."""
        years_of_service = user_profile.get("years_of_service", 0)
        rank = user_profile.get("rank", "").lower()
        
        # Base experience value
        experience_score = min(years_of_service / 20, 0.8)  # Max 0.8 for 20+ years
        
        # Industry-specific bonuses
        if industry in ["defense_aerospace", "government_public_sector", "cybersecurity"]:
            experience_score += 0.3  # Military experience is highly valued
        elif industry in ["project_management", "logistics_supply_chain"]:
            experience_score += 0.2  # Military planning/logistics experience valued
        elif industry in ["technology", "consulting"]:
            experience_score += 0.1  # Some value for discipline and problem-solving
        
        # Leadership experience bonus
        if "officer" in rank or "commander" in rank:
            experience_score += 0.1
        
        return min(1.0, experience_score)
    
    def _get_industry_fit_reasons(self, skill_analysis: Dict[str, Any], industry: str, industry_data: Dict[str, Any]) -> List[str]:
        """Get reasons why the industry is a good fit."""
        reasons = []
        
        # Skill-based reasons
        matching_skills = []
        user_skills = set(skill.lower() for skill in 
                         skill_analysis["technical_skills"] + 
                         skill_analysis["leadership_skills"] + 
                         skill_analysis["domain_skills"])
        industry_skills = set(skill.lower() for skill in industry_data["skill_demand"])
        
        matching_skills = list(user_skills & industry_skills)
        if matching_skills:
            reasons.append(f"Strong skill match in: {', '.join(matching_skills[:3])}")
        
        # Leadership experience
        if skill_analysis["leadership_skills"]:
            reasons.append("Leadership experience highly valued")
        
        # Industry characteristics
        if industry_data["veteran_suitability"] > 0.85:
            reasons.append("Industry known for hiring veterans")
        
        if industry_data["job_security"] > 0.8:
            reasons.append("High job security and stability")
