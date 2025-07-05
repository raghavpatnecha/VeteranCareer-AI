"""
Military skill translation module for converting military experience to civilian equivalents.
Provides AI-powered skill mapping and industry alignment for veterans.
"""

import spacy
import json
import csv
import io
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class SkillTranslation:
    """Data class for military skill translation results."""
    military_skill: str
    civilian_equivalent: str
    confidence_score: float
    applicable_industries: List[str]
    related_job_titles: List[str]
    skill_category: str

class MilitarySkillTranslator:
    """
    AI-powered translator for converting military skills to civilian equivalents.
    Uses NLP and ML techniques for accurate skill mapping.
    """
    
    def __init__(self):
        """Initialize the skill translator with NLP models and skill mappings."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using fallback methods.")
            self.nlp = None
            
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize skill mappings
        self._initialize_skill_mappings()
        self._initialize_industry_mappings()
        self._initialize_job_title_mappings()
        
        # Fit vectorizer with skill corpus
        self._prepare_vectorizer()
    
    def _initialize_skill_mappings(self):
        """Initialize comprehensive military to civilian skill mappings."""
        self.skill_mappings = {
            # Leadership Skills
            "squad leader": {
                "civilian": "Team Leader",
                "category": "Leadership",
                "confidence": 0.95,
                "industries": ["Management", "Operations", "Sales"],
                "job_titles": ["Team Leader", "Supervisor", "Operations Manager"]
            },
            "platoon commander": {
                "civilian": "Project Manager",
                "category": "Leadership",
                "confidence": 0.90,
                "industries": ["Management", "Construction", "Engineering"],
                "job_titles": ["Project Manager", "Operations Manager", "Department Head"]
            },
            "company commander": {
                "civilian": "Director",
                "category": "Leadership",
                "confidence": 0.88,
                "industries": ["Management", "Operations", "Executive"],
                "job_titles": ["Director", "General Manager", "Executive Manager"]
            },
            
            # Technical Skills
            "radio operator": {
                "civilian": "Communications Technician",
                "category": "Technical",
                "confidence": 0.92,
                "industries": ["Telecommunications", "Broadcasting", "Emergency Services"],
                "job_titles": ["Communications Technician", "Radio Operator", "Dispatcher"]
            },
            "electronics technician": {
                "civilian": "Electronics Technician",
                "category": "Technical",
                "confidence": 0.95,
                "industries": ["Electronics", "Manufacturing", "Telecommunications"],
                "job_titles": ["Electronics Technician", "Field Service Technician", "Electronics Engineer"]
            },
            "aircraft mechanic": {
                "civilian": "Aviation Maintenance Technician",
                "category": "Technical",
                "confidence": 0.98,
                "industries": ["Aviation", "Aerospace", "Transportation"],
                "job_titles": ["Aircraft Mechanic", "Aviation Technician", "Maintenance Supervisor"]
            },
            
            # Security Skills
            "military police": {
                "civilian": "Security Officer",
                "category": "Security",
                "confidence": 0.90,
                "industries": ["Security", "Law Enforcement", "Corporate Security"],
                "job_titles": ["Security Officer", "Loss Prevention Specialist", "Corporate Security Manager"]
            },
            "intelligence analyst": {
                "civilian": "Data Analyst",
                "category": "Analysis",
                "confidence": 0.85,
                "industries": ["Intelligence", "Data Analytics", "Cybersecurity"],
                "job_titles": ["Intelligence Analyst", "Data Analyst", "Security Analyst"]
            },
            
            # Logistics Skills
            "supply sergeant": {
                "civilian": "Supply Chain Manager",
                "category": "Logistics",
                "confidence": 0.88,
                "industries": ["Supply Chain", "Logistics", "Procurement"],
                "job_titles": ["Supply Chain Manager", "Procurement Specialist", "Inventory Manager"]
            },
            "logistics coordinator": {
                "civilian": "Logistics Coordinator",
                "category": "Logistics",
                "confidence": 0.92,
                "industries": ["Logistics", "Transportation", "Supply Chain"],
                "job_titles": ["Logistics Coordinator", "Operations Coordinator", "Supply Chain Analyst"]
            },
            
            # Administrative Skills
            "administrative specialist": {
                "civilian": "Administrative Assistant",
                "category": "Administrative",
                "confidence": 0.90,
                "industries": ["Administration", "Office Management", "Human Resources"],
                "job_titles": ["Administrative Assistant", "Office Manager", "Executive Assistant"]
            },
            "personnel specialist": {
                "civilian": "Human Resources Specialist",
                "category": "Administrative",
                "confidence": 0.88,
                "industries": ["Human Resources", "Personnel Management", "Administration"],
                "job_titles": ["HR Specialist", "Personnel Coordinator", "Recruiter"]
            },
            
            # Medical Skills
            "combat medic": {
                "civilian": "Emergency Medical Technician",
                "category": "Medical",
                "confidence": 0.85,
                "industries": ["Healthcare", "Emergency Services", "Medical"],
                "job_titles": ["EMT", "Paramedic", "Medical Assistant"]
            },
            "hospital corpsman": {
                "civilian": "Medical Assistant",
                "category": "Medical",
                "confidence": 0.82,
                "industries": ["Healthcare", "Medical", "Clinical"],
                "job_titles": ["Medical Assistant", "Clinical Assistant", "Healthcare Technician"]
            },
            
            # Engineering Skills
            "combat engineer": {
                "civilian": "Civil Engineer",
                "category": "Engineering",
                "confidence": 0.80,
                "industries": ["Construction", "Engineering", "Infrastructure"],
                "job_titles": ["Civil Engineer", "Construction Manager", "Project Engineer"]
            },
            "signal specialist": {
                "civilian": "Network Technician",
                "category": "Technical",
                "confidence": 0.85,
                "industries": ["Telecommunications", "IT", "Networking"],
                "job_titles": ["Network Technician", "IT Specialist", "Communications Engineer"]
            }
        }
    
    def _initialize_industry_mappings(self):
        """Initialize industry-specific skill mappings."""
        self.industry_mappings = {
            "Defense": ["Security", "Law Enforcement", "Government"],
            "Technology": ["IT", "Software", "Telecommunications", "Cybersecurity"],
            "Healthcare": ["Medical", "Emergency Services", "Clinical"],
            "Manufacturing": ["Production", "Quality Control", "Operations"],
            "Logistics": ["Supply Chain", "Transportation", "Procurement"],
            "Construction": ["Engineering", "Project Management", "Infrastructure"],
            "Aviation": ["Aerospace", "Transportation", "Maintenance"],
            "Government": ["Public Administration", "Policy", "Compliance"]
        }
    
    def _initialize_job_title_mappings(self):
        """Initialize job title mappings by skill category."""
        self.job_title_mappings = {
            "Leadership": [
                "Team Leader", "Supervisor", "Manager", "Director", "Executive",
                "Project Manager", "Operations Manager", "Department Head"
            ],
            "Technical": [
                "Technician", "Specialist", "Engineer", "Analyst", "Coordinator",
                "Field Service Technician", "Maintenance Technician"
            ],
            "Security": [
                "Security Officer", "Security Manager", "Loss Prevention Specialist",
                "Corporate Security Manager", "Safety Coordinator"
            ],
            "Administrative": [
                "Administrative Assistant", "Office Manager", "Executive Assistant",
                "Coordinator", "Specialist", "Analyst"
            ],
            "Medical": [
                "Medical Assistant", "Healthcare Technician", "EMT", "Paramedic",
                "Clinical Assistant", "Medical Technician"
            ],
            "Logistics": [
                "Supply Chain Manager", "Logistics Coordinator", "Procurement Specialist",
                "Inventory Manager", "Operations Coordinator"
            ],
            "Engineering": [
                "Civil Engineer", "Project Engineer", "Construction Manager",
                "Systems Engineer", "Design Engineer"
            ],
            "Analysis": [
                "Data Analyst", "Business Analyst", "Intelligence Analyst",
                "Research Analyst", "Operations Analyst"
            ]
        }
    
    def _prepare_vectorizer(self):
        """Prepare TF-IDF vectorizer with skill corpus."""
        # Create corpus from military skills and civilian equivalents
        corpus = []
        for skill, mapping in self.skill_mappings.items():
            corpus.append(skill)
            corpus.append(mapping["civilian"])
        
        # Add additional skill terms
        additional_skills = [
            "team management", "project coordination", "problem solving",
            "communication", "technical expertise", "leadership development",
            "training and development", "quality assurance", "risk management",
            "strategic planning", "operational efficiency", "customer service"
        ]
        corpus.extend(additional_skills)
        
        try:
            self.vectorizer.fit(corpus)
            self.skill_vectors = self.vectorizer.transform(corpus)
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            self.skill_vectors = None
    
    def translate_skill(self, military_skill: str) -> SkillTranslation:
        """
        Translate a single military skill to civilian equivalent.
        
        Args:
            military_skill: Military skill or experience description
            
        Returns:
            SkillTranslation object with civilian equivalent and metadata
        """
        # Normalize input
        normalized_skill = military_skill.lower().strip()
        
        # Check direct mappings first
        if normalized_skill in self.skill_mappings:
            mapping = self.skill_mappings[normalized_skill]
            return SkillTranslation(
                military_skill=military_skill,
                civilian_equivalent=mapping["civilian"],
                confidence_score=mapping["confidence"],
                applicable_industries=mapping["industries"],
                related_job_titles=mapping["job_titles"],
                skill_category=mapping["category"]
            )
        
        # Use semantic similarity for unknown skills
        return self._semantic_skill_matching(military_skill)
    
    def _semantic_skill_matching(self, military_skill: str) -> SkillTranslation:
        """Use semantic similarity to find best civilian equivalent."""
        if self.skill_vectors is None:
            return self._fallback_translation(military_skill)
        
        try:
            # Vectorize input skill
            skill_vector = self.vectorizer.transform([military_skill.lower()])
            
            # Calculate similarities with known skills
            similarities = cosine_similarity(skill_vector, self.skill_vectors).flatten()
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity > 0.3:  # Threshold for acceptable similarity
                # Get corresponding skill mapping
                skill_keys = list(self.skill_mappings.keys())
                if best_match_idx < len(skill_keys):
                    matched_skill = skill_keys[best_match_idx]
                    mapping = self.skill_mappings[matched_skill]
                    
                    return SkillTranslation(
                        military_skill=military_skill,
                        civilian_equivalent=mapping["civilian"],
                        confidence_score=best_similarity * mapping["confidence"],
                        applicable_industries=mapping["industries"],
                        related_job_titles=mapping["job_titles"],
                        skill_category=mapping["category"]
                    )
        
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
        
        return self._fallback_translation(military_skill)
    
    def _fallback_translation(self, military_skill: str) -> SkillTranslation:
        """Provide fallback translation for unknown skills."""
        # Extract keywords and provide generic translation
        keywords = self._extract_keywords(military_skill)
        
        # Determine category based on keywords
        category = self._categorize_skill(keywords)
        
        # Generate generic civilian equivalent
        civilian_equivalent = self._generate_civilian_equivalent(military_skill, category)
        
        return SkillTranslation(
            military_skill=military_skill,
            civilian_equivalent=civilian_equivalent,
            confidence_score=0.5,  # Lower confidence for fallback
            applicable_industries=self._get_industries_for_category(category),
            related_job_titles=self._get_job_titles_for_category(category),
            skill_category=category
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from skill description."""
        if self.nlp:
            doc = self.nlp(text.lower())
            keywords = [token.lemma_ for token in doc 
                       if not token.is_stop and not token.is_punct and token.is_alpha]
        else:
            # Fallback without spaCy
            keywords = [word.lower() for word in text.split() 
                       if len(word) > 2 and word.lower() not in ['the', 'and', 'or', 'in', 'on', 'at']]
        
        return keywords
    
    def _categorize_skill(self, keywords: List[str]) -> str:
        """Categorize skill based on keywords."""
        category_keywords = {
            "Leadership": ["leader", "command", "manage", "supervise", "direct", "coordinate"],
            "Technical": ["technical", "equipment", "system", "maintenance", "repair", "operate"],
            "Security": ["security", "guard", "protect", "patrol", "surveillance", "safety"],
            "Medical": ["medical", "health", "care", "treatment", "emergency", "first"],
            "Administrative": ["admin", "office", "paperwork", "record", "document", "clerk"],
            "Logistics": ["supply", "logistics", "transport", "inventory", "procurement", "distribution"],
            "Analysis": ["analyze", "intelligence", "data", "research", "investigate", "assess"]
        }
        
        for category, cat_keywords in category_keywords.items():
            if any(keyword in keywords for keyword in cat_keywords):
                return category
        
        return "General"
    
    def _generate_civilian_equivalent(self, military_skill: str, category: str) -> str:
        """Generate civilian equivalent based on category."""
        category_mappings = {
            "Leadership": "Team Leader/Supervisor",
            "Technical": "Technical Specialist",
            "Security": "Security Specialist",
            "Medical": "Healthcare Technician",
            "Administrative": "Administrative Specialist",
            "Logistics": "Logistics Coordinator",
            "Analysis": "Data Analyst",
            "General": "Professional Specialist"
        }
        
        return category_mappings.get(category, "Professional Specialist")
    
    def _get_industries_for_category(self, category: str) -> List[str]:
        """Get applicable industries for skill category."""
        return self.job_title_mappings.get(category, ["General Business", "Operations"])
    
    def _get_job_titles_for_category(self, category: str) -> List[str]:
        """Get related job titles for skill category."""
        return self.job_title_mappings.get(category, ["Specialist", "Coordinator", "Associate"])
    
    def translate_multiple_skills(self, military_skills: List[str]) -> List[SkillTranslation]:
        """
        Translate multiple military skills to civilian equivalents.
        
        Args:
            military_skills: List of military skills or experiences
            
        Returns:
            List of SkillTranslation objects
        """
        translations = []
        for skill in military_skills:
            try:
                translation = self.translate_skill(skill)
                translations.append(translation)
            except Exception as e:
                logger.error(f"Error translating skill '{skill}': {e}")
                # Add fallback translation
                translations.append(self._fallback_translation(skill))
        
        return translations
    
    def get_skill_recommendations(self, user_profile: Dict) -> List[Dict]:
        """
        Get skill development recommendations based on user profile.
        
        Args:
            user_profile: User profile containing military background
            
        Returns:
            List of skill development recommendations
        """
        recommendations = []
        
        # Extract military skills from profile
        military_background = user_profile.get('military_background', {})
        branch = military_background.get('branch', '').lower()
        rank = military_background.get('rank', '').lower()
        mos = military_background.get('mos', '').lower()
        
        # Branch-specific recommendations
        branch_recommendations = {
            'army': ['Leadership Development', 'Project Management', 'Logistics Management'],
            'navy': ['Operations Management', 'Technical Systems', 'Maritime Operations'],
            'air force': ['Systems Management', 'Technical Operations', 'Quality Control'],
            'marines': ['Team Leadership', 'Operations Planning', 'Security Management']
        }
        
        # Rank-based recommendations
        if any(title in rank for title in ['officer', 'lieutenant', 'captain', 'major']):
            recommendations.extend([
                {'skill': 'Strategic Planning', 'priority': 'High', 'category': 'Leadership'},
                {'skill': 'Executive Management', 'priority': 'High', 'category': 'Leadership'},
                {'skill': 'Budget Management', 'priority': 'Medium', 'category': 'Administrative'}
            ])
        elif any(title in rank for title in ['sergeant', 'corporal', 'specialist']):
            recommendations.extend([
                {'skill': 'Team Supervision', 'priority': 'High', 'category': 'Leadership'},
                {'skill': 'Technical Training', 'priority': 'Medium', 'category': 'Technical'},
                {'skill': 'Quality Assurance', 'priority': 'Medium', 'category': 'Technical'}
            ])
        
        # Add branch-specific recommendations
        if branch in branch_recommendations:
            for skill in branch_recommendations[branch]:
                recommendations.append({
                    'skill': skill,
                    'priority': 'Medium',
                    'category': self._categorize_skill([skill.lower()])
                })
        
        return recommendations
    
    def export_translations(self, translations: List[SkillTranslation], format: str = 'json') -> str:
        """
        Export skill translations in specified format.
        
        Args:
            translations: List of skill translations
            format: Export format ('json' or 'csv')
            
        Returns:
            Formatted export string
        """
        if format.lower() == 'json':
            export_data = []
            for t in translations:
                export_data.append({
                    'military_skill': t.military_skill,
                    'civilian_equivalent': t.civilian_equivalent,
                    'confidence_score': t.confidence_score,
                    'applicable_industries': t.applicable_industries,
                    'related_job_titles': t.related_job_titles,
                    'skill_category': t.skill_category
                })
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Military Skill', 'Civilian Equivalent', 'Confidence Score', 'Industries', 'Job Titles'
            ])

            # Write data rows
            for t in translations:
                writer.writerow([
                    t.military_skill,
                    t.civilian_equivalent,
                    t.confidence_score,
                    ', '.join(t.applicable_industries),
                    ', '.join(t.related_job_titles)
                ])

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {format}")