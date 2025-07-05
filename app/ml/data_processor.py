import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import string
from collections import Counter, defaultdict
import unicodedata
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    clean_text: bool = True
    normalize_skills: bool = True
    extract_features: bool = True
    handle_missing: bool = True
    remove_duplicates: bool = True
    standardize_locations: bool = True
    normalize_salaries: bool = True
    extract_experience: bool = True
    min_text_length: int = 10
    max_text_length: int = 10000
    skill_min_frequency: int = 2
    location_normalization_threshold: float = 0.8


@dataclass
class ProcessingResult:
    """Result of data processing operations."""
    processed_data: pd.DataFrame
    feature_matrices: Dict[str, Any]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_log: List[str]
    errors: List[str]


class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    def __init__(self):
        self.nlp = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {str(e)}")
            self.stop_words = set()
        
        # Load spaCy model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy model not available: {str(e)}")
    
    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """Clean and normalize text content."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\(\)]', ' ', text)
        
        # Convert to lowercase if not preserving case
        if not preserve_case:
            text = text.lower()
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 50) -> List[str]:
        """Extract keywords from text using NLP techniques."""
        if not text:
            return []
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            keywords = []
            
            for token in doc:
                if (token.is_alpha and 
                    not token.is_stop and 
                    len(token.text) > 2 and
                    token.pos_ in ['NOUN', 'ADJ', 'PROPN']):
                    keywords.append(token.lemma_.lower())
            
            # Add noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep short phrases
                    keywords.append(chunk.text.lower())
        
        else:
            # Fallback to NLTK
            try:
                tokens = word_tokenize(text.lower())
                keywords = [token for token in tokens 
                           if token.isalpha() and 
                           token not in self.stop_words and 
                           len(token) > 2]
            except:
                keywords = text.lower().split()
        
        # Remove duplicates and return top keywords
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(max_keywords)]
    
    def normalize_skill_text(self, skill: str) -> str:
        """Normalize skill descriptions for consistency."""
        if not skill:
            return ""
        
        skill = skill.strip().lower()
        
        # Common skill normalizations
        skill_mappings = {
            # Programming languages
            'javascript': 'javascript',
            'js': 'javascript',
            'node.js': 'nodejs',
            'node': 'nodejs',
            'react.js': 'react',
            'reactjs': 'react',
            'vue.js': 'vue',
            'vuejs': 'vue',
            'angular.js': 'angular',
            'angularjs': 'angular',
            
            # Databases
            'mysql': 'mysql',
            'my sql': 'mysql',
            'postgresql': 'postgresql',
            'postgres': 'postgresql',
            'mongo db': 'mongodb',
            'mongo': 'mongodb',
            
            # Cloud platforms
            'amazon web services': 'aws',
            'amazon aws': 'aws',
            'microsoft azure': 'azure',
            'google cloud platform': 'gcp',
            'google cloud': 'gcp',
            
            # General tech
            'artificial intelligence': 'ai',
            'machine learning': 'ml',
            'deep learning': 'deep learning',
            'natural language processing': 'nlp',
            'user experience': 'ux',
            'user interface': 'ui',
            'search engine optimization': 'seo',
            
            # Business skills
            'project management': 'project management',
            'project mgmt': 'project management',
            'people management': 'people management',
            'team management': 'team management',
            'team lead': 'team leadership',
            'team leader': 'team leadership',
            'team leading': 'team leadership',
            
            # Military skills
            'squad leader': 'team leadership',
            'platoon commander': 'operations management',
            'company commander': 'senior management',
            'logistics coordinator': 'logistics management',
            'supply chain': 'supply chain management',
            'quality assurance': 'quality control',
            'quality control': 'quality control',
            
            # Certifications
            'pmp certified': 'pmp',
            'pmp certification': 'pmp',
            'six sigma': 'six sigma',
            'lean six sigma': 'lean six sigma',
            'agile certified': 'agile',
            'scrum master': 'scrum master',
            'certified scrum master': 'scrum master'
        }
        
        # Apply mappings
        for pattern, replacement in skill_mappings.items():
            if pattern in skill:
                skill = skill.replace(pattern, replacement)
        
        # Remove common suffixes/prefixes
        prefixes_to_remove = ['certified', 'advanced', 'basic', 'intermediate', 'expert']
        suffixes_to_remove = ['certification', 'certified', 'experience', 'skills', 'knowledge']
        
        words = skill.split()
        words = [word for word in words 
                if word not in prefixes_to_remove and word not in suffixes_to_remove]
        
        return ' '.join(words).strip()


class LocationNormalizer:
    """Location data normalization and standardization."""
    
    def __init__(self):
        self.location_mappings = self._load_location_mappings()
        self.city_aliases = self._load_city_aliases()
    
    def _load_location_mappings(self) -> Dict[str, str]:
        """Load standard location mappings."""
        return {
            # NCR Region
            'new delhi': 'delhi',
            'delhi ncr': 'delhi',
            'ncr': 'delhi',
            'gurgaon': 'gurgaon',
            'gurugram': 'gurgaon',
            'noida': 'noida',
            'faridabad': 'faridabad',
            'ghaziabad': 'ghaziabad',
            
            # Mumbai Region
            'bombay': 'mumbai',
            'navi mumbai': 'mumbai',
            'thane': 'mumbai',
            'pune': 'pune',
            'poona': 'pune',
            
            # Bangalore Region
            'bengaluru': 'bangalore',
            'bangalore': 'bangalore',
            'whitefield': 'bangalore',
            'electronic city': 'bangalore',
            
            # Chennai Region
            'madras': 'chennai',
            'chennai': 'chennai',
            
            # Hyderabad Region
            'hyderabad': 'hyderabad',
            'secunderabad': 'hyderabad',
            'cyberabad': 'hyderabad',
            
            # Kolkata Region
            'calcutta': 'kolkata',
            'kolkata': 'kolkata',
            
            # Other major cities
            'ahmedabad': 'ahmedabad',
            'amdavad': 'ahmedabad',
            'jaipur': 'jaipur',
            'lucknow': 'lucknow',
            'kanpur': 'kanpur',
            'nagpur': 'nagpur',
            'indore': 'indore',
            'bhopal': 'bhopal',
            'visakhapatnam': 'visakhapatnam',
            'vizag': 'visakhapatnam',
            'vadodara': 'vadodara',
            'baroda': 'vadodara',
            'ludhiana': 'ludhiana',
            'agra': 'agra',
            'nashik': 'nashik',
            'meerut': 'meerut',
            'rajkot': 'rajkot',
            'varanasi': 'varanasi',
            'benares': 'varanasi',
            'srinagar': 'srinagar',
            'jodhpur': 'jodhpur',
            'amritsar': 'amritsar',
            'raipur': 'raipur',
            'allahabad': 'allahabad',
            'prayagraj': 'allahabad',
            'coimbatore': 'coimbatore',
            'jabalpur': 'jabalpur',
            'gwalior': 'gwalior',
            'vijayawada': 'vijayawada',
            'madurai': 'madurai',
            'guwahati': 'guwahati',
            'chandigarh': 'chandigarh',
            'mysore': 'mysore',
            'mysuru': 'mysore',
            'thiruvananthapuram': 'thiruvananthapuram',
            'trivandrum': 'thiruvananthapuram',
            'kochi': 'kochi',
            'cochin': 'kochi'
        }
    
    def _load_city_aliases(self) -> Dict[str, List[str]]:
        """Load city aliases for fuzzy matching."""
        return {
            'delhi': ['new delhi', 'delhi ncr', 'ncr', 'national capital region'],
            'mumbai': ['bombay', 'navi mumbai', 'greater mumbai'],
            'bangalore': ['bengaluru', 'silicon valley of india'],
            'chennai': ['madras'],
            'hyderabad': ['secunderabad', 'cyberabad', 'hitec city'],
            'kolkata': ['calcutta'],
            'ahmedabad': ['amdavad'],
            'pune': ['poona'],
            'vadodara': ['baroda'],
            'visakhapatnam': ['vizag'],
            'thiruvananthapuram': ['trivandrum'],
            'kochi': ['cochin'],
            'allahabad': ['prayagraj'],
            'varanasi': ['benares'],
            'mysore': ['mysuru']
        }
    
    def normalize_location(self, location: str) -> str:
        """Normalize location string to standard format."""
        if not location:
            return ""
        
        location = location.strip().lower()
        
        # Remove common location indicators
        location = re.sub(r'\b(city|town|district|state|region|area)\b', '', location)
        location = re.sub(r'\s+', ' ', location).strip()
        
        # Direct mapping
        if location in self.location_mappings:
            return self.location_mappings[location]
        
        # Check for partial matches
        for standard_city, aliases in self.city_aliases.items():
            if location in aliases or any(alias in location for alias in aliases):
                return standard_city
        
        # Check if location contains a known city
        for mapped_location, standard_location in self.location_mappings.items():
            if mapped_location in location:
                return standard_location
        
        # Return cleaned location if no mapping found
        return location.title()


class SalaryNormalizer:
    """Salary data normalization and standardization."""
    
    def __init__(self):
        self.currency_symbols = {'₹', 'rs', 'inr', 'rupees', 'rupee', '$', 'usd', 'dollar'}
        self.time_periods = {'month', 'monthly', 'year', 'yearly', 'annual', 'annum', 'pa', 'per annum'}
    
    def normalize_salary(self, salary_text: str) -> Dict[str, Optional[int]]:
        """
        Normalize salary text to structured format.
        
        Returns:
            Dict with 'min', 'max', 'currency', 'period' keys
        """
        if not salary_text or not isinstance(salary_text, str):
            return {'min': None, 'max': None, 'currency': 'INR', 'period': 'annual'}
        
        salary_text = salary_text.lower().strip()
        
        # Skip if not disclosed
        if any(phrase in salary_text for phrase in ['not disclosed', 'not specified', 'confidential', 'competitive']):
            return {'min': None, 'max': None, 'currency': 'INR', 'period': 'annual'}
        
        # Extract currency
        currency = 'INR'  # Default to INR
        if any(symbol in salary_text for symbol in ['$', 'usd', 'dollar']):
            currency = 'USD'
        
        # Extract time period
        period = 'annual'  # Default to annual
        if any(term in salary_text for term in ['month', 'monthly']):
            period = 'monthly'
        
        # Extract numbers
        numbers = self._extract_salary_numbers(salary_text)
        
        if not numbers:
            return {'min': None, 'max': None, 'currency': currency, 'period': period}
        
        # Determine min and max
        if len(numbers) == 1:
            # Single number - assume it's either min or exact
            if any(term in salary_text for term in ['upto', 'up to', 'maximum', 'max']):
                return {'min': None, 'max': numbers[0], 'currency': currency, 'period': period}
            elif any(term in salary_text for term in ['minimum', 'min', 'starting', 'from']):
                return {'min': numbers[0], 'max': None, 'currency': currency, 'period': period}
            else:
                return {'min': numbers[0], 'max': numbers[0], 'currency': currency, 'period': period}
        else:
            # Multiple numbers - take min and max
            return {'min': min(numbers), 'max': max(numbers), 'currency': currency, 'period': period}
    
    def _extract_salary_numbers(self, text: str) -> List[int]:
        """Extract salary numbers from text."""
        numbers = []
        
        # Pattern for Indian salary format (lakhs, crores)
        # Examples: 5L, 10 lakh, 1.5 crore, 12.5L
        lakh_pattern = r'(\d+(?:\.\d+)?)\s*(?:l|lakh|lakhs?)\b'
        crore_pattern = r'(\d+(?:\.\d+)?)\s*(?:cr|crore|crores?)\b'
        
        # Find lakhs
        for match in re.finditer(lakh_pattern, text, re.IGNORECASE):
            value = float(match.group(1)) * 100000  # Convert lakhs to rupees
            numbers.append(int(value))
        
        # Find crores
        for match in re.finditer(crore_pattern, text, re.IGNORECASE):
            value = float(match.group(1)) * 10000000  # Convert crores to rupees
            numbers.append(int(value))
        
        # Pattern for regular numbers (thousands, millions)
        # Examples: 50000, 5,00,000, 1,200,000
        number_pattern = r'(\d{1,3}(?:,\d{2,3})*|\d+)'
        
        for match in re.finditer(number_pattern, text):
            number_str = match.group(1).replace(',', '')
            try:
                number = int(number_str)
                
                # Filter reasonable salary ranges (10K to 100Cr in INR)
                if 10000 <= number <= 1000000000:
                    # If number is less than 10000, assume it's in thousands
                    if number < 10000:
                        number *= 1000
                    numbers.append(number)
            except ValueError:
                continue
        
        return sorted(list(set(numbers)))


class ExperienceExtractor:
    """Extract and normalize experience requirements from text."""
    
    def extract_experience(self, text: str) -> Dict[str, Optional[int]]:
        """
        Extract experience requirements from job text.
        
        Returns:
            Dict with 'min_years', 'max_years', 'level' keys
        """
        if not text:
            return {'min_years': None, 'max_years': None, 'level': None}
        
        text = text.lower()
        
        # Patterns for experience extraction
        patterns = [
            r'(\d+)[-\s]*(?:to|-)[-\s]*(\d+)\s*(?:years?|yrs?)',  # 3-5 years, 2 to 4 years
            r'(\d+)\+\s*(?:years?|yrs?)',  # 5+ years
            r'minimum\s*(\d+)\s*(?:years?|yrs?)',  # minimum 3 years
            r'maximum\s*(\d+)\s*(?:years?|yrs?)',  # maximum 5 years
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',  # 3 years experience
            r'experience\s*(?:of\s*)?(\d+)[-\s]*(?:to|-)[-\s]*(\d+)\s*(?:years?|yrs?)',  # experience of 2-4 years
        ]
        
        min_years = None
        max_years = None
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:  # Range pattern
                    min_years = int(groups[0])
                    max_years = int(groups[1])
                    break
                elif len(groups) == 1:  # Single number
                    if 'minimum' in match.group(0):
                        min_years = int(groups[0])
                    elif 'maximum' in match.group(0):
                        max_years = int(groups[0])
                    elif '+' in match.group(0):
                        min_years = int(groups[0])
                    else:
                        # Assume it's exact or minimum
                        min_years = int(groups[0])
                        max_years = int(groups[0])
                    break
        
        # Determine experience level
        level = self._determine_experience_level(min_years, max_years, text)
        
        return {'min_years': min_years, 'max_years': max_years, 'level': level}
    
    def _determine_experience_level(self, min_years: Optional[int], max_years: Optional[int], text: str) -> Optional[str]:
        """Determine experience level from years and text context."""
        # Check for explicit level mentions
        if any(term in text for term in ['fresher', 'graduate', 'entry level', 'junior']):
            return 'entry'
        elif any(term in text for term in ['senior', 'lead', 'principal', 'architect']):
            return 'senior'
        elif any(term in text for term in ['manager', 'director', 'head', 'vp', 'executive']):
            return 'executive'
        
        # Determine based on years
        avg_years = None
        if min_years and max_years:
            avg_years = (min_years + max_years) / 2
        elif min_years:
            avg_years = min_years
        elif max_years:
            avg_years = max_years
        
        if avg_years is not None:
            if avg_years <= 2:
                return 'entry'
            elif avg_years <= 7:
                return 'mid'
            elif avg_years <= 12:
                return 'senior'
            else:
                return 'executive'
        
        return None


class JobDataProcessor:
    """Main data processing class for job data cleaning and normalization."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.text_cleaner = TextCleaner()
        self.location_normalizer = LocationNormalizer()
        self.salary_normalizer = SalaryNormalizer()
        self.experience_extractor = ExperienceExtractor()
        
        # Processing statistics
        self.processing_stats = {
            'records_processed': 0,
            'records_cleaned': 0,
            'duplicates_removed': 0,
            'missing_data_handled': 0,
            'errors_encountered': 0
        }
    
    def process_job_data(self, data: pd.DataFrame) -> ProcessingResult:
        """
        Process job data with comprehensive cleaning and normalization.
        
        Args:
            data: Raw job data DataFrame
            
        Returns:
            ProcessingResult with processed data and metadata
        """
        try:
            processing_log = []
            errors = []
            
            processing_log.append(f"Starting processing of {len(data)} records")
            
            # Create a copy to avoid modifying original data
            processed_data = data.copy()
            
            # 1. Basic data cleaning
            if self.config.clean_text:
                processed_data = self._clean_text_fields(processed_data, processing_log)
            
            # 2. Handle missing values
            if self.config.handle_missing:
                processed_data = self._handle_missing_values(processed_data, processing_log)
            
            # 3. Remove duplicates
            if self.config.remove_duplicates:
                processed_data = self._remove_duplicates(processed_data, processing_log)
            
            # 4. Normalize locations
            if self.config.standardize_locations:
                processed_data = self._normalize_locations(processed_data, processing_log)
            
            # 5. Normalize salaries
            if self.config.normalize_salaries:
                processed_data = self._normalize_salaries(processed_data, processing_log)
            
            # 6. Extract experience information
            if self.config.extract_experience:
                processed_data = self._extract_experience_info(processed_data, processing_log)
            
            # 7. Normalize skills
            if self.config.normalize_skills:
                processed_data = self._normalize_skills(processed_data, processing_log)
            
            # 8. Extract features for ML
            feature_matrices = {}
            if self.config.extract_features:
                feature_matrices = self._extract_ml_features(processed_data, processing_log)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(data, processed_data)
            
            # Create metadata
            metadata = {
                'original_records': len(data),
                'processed_records': len(processed_data),
                'processing_timestamp': datetime.now().isoformat(),
                'config_used': self.config.__dict__,
                'columns_processed': list(processed_data.columns),
                'processing_stats': self.processing_stats
            }
            
            processing_log.append("Processing completed successfully")
            
            return ProcessingResult(
                processed_data=processed_data,
                feature_matrices=feature_matrices,
                metadata=metadata,
                quality_metrics=quality_metrics,
                processing_log=processing_log,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error in job data processing: {str(e)}")
            errors.append(f"Processing failed: {str(e)}")
            
            return ProcessingResult(
                processed_data=pd.DataFrame(),
                feature_matrices={},
                metadata={'error': str(e)},
                quality_metrics={},
                processing_log=processing_log,
                errors=errors
            )
    
    def _clean_text_fields(self, data: pd.DataFrame, log: List[str]) -> pd.DataFrame:
        """Clean text fields in the dataset."""
        text_columns = ['title', 'description', 'requirements', 'company_name']
        
        for col in text_columns:
            if col in data.columns:
                original_count = data[col].notna().sum()
                data[col] = data[col].apply(
                    lambda x: self.text_cleaner.clean_text(x) if pd.notna(x) else x
                )
                # Remove entries that became empty after cleaning
                empty_mask = data[col].str.len() < self.config.min_text_length
                data.loc[empty_mask, col] = np.nan
                
                cleaned_count = data[col].notna().sum()
                log.append(f"Cleaned {col}: {original_count} -> {cleaned_count} valid entries")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, log: List[str]) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_before = data.isnull().sum().sum()
        
        # Fill missing values with appropriate defaults
        default_values = {
            'job_type': 'full_time',
            'experience_level': 'not_specified',
            'is_remote': False,
            'veteran_preference': False,
            'government_job': False,
            'psu_job': False,
            'is_active': True,
            'view_count': 0,
            'veteran_match_score': 0.0
        }
        
        for col, default_val in default_values.items():
            if col in data.columns:
                data[col] = data[col].fillna(default_val)
        
        # For text fields, keep NaN to indicate missing data
        # For numerical fields, use median or mode
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col not in default_values and data[col].isnull().any():
                if col.endswith('_score') or col.endswith('_count'):
                    data[col] = data[col].fillna(0)
                else:
                    median_val = data[col].median()
                    if pd.notna(median_val):
                        data[col] = data[col].fillna(median_val)
        
        missing_after = data.isnull().sum().sum()
        self.processing_stats['missing_data_handled'] = missing_before - missing_after
        log.append(f"Handled missing values: {missing_before} -> {missing_after}")
        
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame, log: List[str]) -> pd.DataFrame:
        """Remove duplicate job entries."""
        original_count = len(data)
        
        # Define columns for duplicate detection
        duplicate_columns = ['title', 'company_name', 'location']
        available_columns = [col for col in duplicate_columns if col in data.columns]
        
        if available_columns:
            data = data.drop_duplicates(subset=available_columns, keep='first')
        
        duplicates_removed = original_count - len(data)
        self.processing_stats['duplicates_removed'] = duplicates_removed
        log.append(f"Removed {duplicates_removed} duplicate records")
        
        return data
    
    def _normalize_locations(self, data: pd.DataFrame, log: List[str]) -> pd.DataFrame:
        """Normalize location data."""
        if 'location' in data.columns:
            original_unique = data['location'].nunique()
            
            data['location_normalized'] = data['location'].apply(
                lambda x: self.location_normalizer.normalize_location(x) if pd.notna(x) else x
            )
            
            normalized_unique = data['location_normalized'].nunique()
            log.append(f"Location normalization: {original_unique} -> {normalized_unique} unique locations")
        
        return data
    
    def _normalize_salaries(self, data: pd.DataFrame, log: List[str]) -> pd.DataFrame:
        """Normalize salary information."""
        salary_columns = ['salary', 'salary_range', 'compensation']
        
        for col in salary_columns:
            if col in data.columns:
                salary_data = data[col].apply(
                    lambda x: self.salary_normalizer.normalize_salary(x) if pd.notna(x) else {}
                )
                
                # Extract normalized salary components
                data[f'{col}_min'] = salary_data.apply(lambda x: x.get('min'))
                data[f'{col}_max'] = salary_data.apply(lambda x: x.get('max'))
                data[f'{col}_currency'] = salary_data.apply(lambda x: x.get('currency', 'INR'))
                data[f'{col}_period'] = salary_data.apply(lambda x: x.get('period', 'annual'))
                
                # Standardize to annual INR for comparison
                annual_salaries = []
                for _, row in data.iterrows():
                    min_sal = row.get(f'{col}_
