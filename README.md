# VeteranCareer AI

AI-powered job discovery and career guidance platform designed exclusively for Indian ex-servicemen and government officers transitioning to civilian careers.

## 🚀 Features

- **Intelligent Job Matching**: AI-powered job recommendations based on military/government experience
- **Skill Translation**: Converts military skills to civilian job market terminology
- **Automated Job Discovery**: Web scraping from major Indian job portals and PSU websites
- **Career Guidance**: Personalized career advice using machine learning
- **Application Tracking**: Comprehensive job application management system
- **Profile Intelligence**: Smart profile optimization recommendations
- **Real-time Notifications**: Email and system notifications for job opportunities

## 📋 Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher
- Redis 6 or higher
- Node.js (for Playwright browser automation)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd veterancareer-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
playwright install chromium
```

### 4. Install Additional System Dependencies

```bash
# PostgreSQL and Redis (if not already installed)
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib redis-server

# macOS:
brew install postgresql redis
```

## ⚙️ Environment Configuration

### 1. Create Environment File

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:your_password@127.0.0.1:5432/veterancareer_db
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=veterancareer_db
DB_USER=postgres
DB_PASSWORD=your_password

# Redis Configuration
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0

# JWT Configuration
JWT_SECRET_KEY=your_super_secret_jwt_key_change_this_in_production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI/ML Configuration
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_FROM=noreply@veterancareer.ai
```

## 🗄️ Database Setup

### 1. Create Database

```bash
# Using createdb command
createdb -h 127.0.0.1 -p 5432 -U postgres veterancareer_db

# Or using psql
psql -h 127.0.0.1 -p 5432 -U postgres -c "CREATE DATABASE veterancareer_db;"
```

### 2. Run Database Migrations

```bash
# Generate initial migration (if needed)
alembic revision --autogenerate -m "Create initial database schema"

# Apply migrations
alembic upgrade head
```

### 3. Verify Database Connection

```bash
python -c "from app.database import engine; from sqlalchemy import text; conn = engine.connect(); result = conn.execute(text('SELECT 1')); print('Database connection successful:', result.fetchone())"
```

## 🚀 Running the Application

### 1. Start Redis Server

```bash
redis-server
```

### 2. Start Celery Worker (Background Tasks)

```bash
celery -A app.celery_app worker --loglevel=info
```

### 3. Start Celery Beat (Scheduled Tasks)

```bash
celery -A app.celery_app beat --loglevel=info
```

### 4. Start FastAPI Application

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Access the Application

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Main Endpoint: http://localhost:8000/

## 📚 API Documentation

### Authentication Endpoints

#### Register User
```bash
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "John Doe",
  "service_branch": "Army",
  "rank": "Major",
  "years_of_service": 15,
  "phone": "+91-9876543210"
}
```

#### Login
```bash
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=secure_password
```

### Profile Endpoints

#### Get Profile
```bash
GET /api/profile/me
Authorization: Bearer <access_token>
```

#### Update Profile
```bash
PUT /api/profile/me
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "skills": ["Leadership", "Project Management", "Strategic Planning"],
  "location": "New Delhi",
  "preferred_locations": ["Mumbai", "Bangalore", "Pune"],
  "salary_expectation": 1500000
}
```

#### Skill Translation
```bash
POST /api/profile/translate-skills
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "military_skills": ["Combat Operations", "Personnel Management", "Logistics Coordination"]
}
```

### Job Endpoints

#### Search Jobs
```bash
GET /api/jobs/search?query=project+manager&location=mumbai&page=1&size=10
Authorization: Bearer <access_token>
```

#### Get Job Recommendations
```bash
GET /api/jobs/recommendations?limit=20
Authorization: Bearer <access_token>
```

#### Get Job Details
```bash
GET /api/jobs/{job_id}
Authorization: Bearer <access_token>
```

### Application Endpoints

#### Apply for Job
```bash
POST /api/applications/apply
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "job_id": "job_uuid",
  "cover_letter": "I am interested in this position...",
  "resume_url": "https://example.com/resume.pdf"
}
```

#### Get Applications
```bash
GET /api/applications/my-applications?status=applied&page=1&size=10
Authorization: Bearer <access_token>
```

#### Update Application Status
```bash
PUT /api/applications/{application_id}/status
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "status": "interview_scheduled",
  "notes": "Interview scheduled for tomorrow"
}
```

## 🏗️ Architecture Overview

### Core Components

1. **FastAPI Application** (`app/main.py`)
   - REST API endpoints
   - Authentication middleware
   - CORS configuration

2. **Database Layer** (`app/models/`)
   - SQLAlchemy ORM models
   - PostgreSQL database
   - Alembic migrations

3. **AI/ML Modules** (`app/ml/`)
   - Skill translation engine
   - Job matching algorithm
   - Career advisory system

4. **Web Scrapers** (`app/scrapers/`)
   - Job portal scrapers
   - PSU website scrapers
   - Rate-limited scraping

5. **Background Tasks** (`app/tasks/`)
   - Celery task queue
   - Redis broker
   - Scheduled job scraping

6. **Authentication** (`app/auth/`)
   - JWT token management
   - Password hashing
   - User session handling

### Data Flow

1. **User Registration/Login** → JWT token generation
2. **Profile Creation** → Skill translation → Civilian skills mapping
3. **Job Scraping** → Background tasks → Database storage
4. **Job Matching** → ML algorithm → Personalized recommendations
5. **Application Tracking** → Status updates → Notifications

## 🔧 Usage Examples

### 1. Complete User Journey

```python
# 1. Register new user
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Register
response = client.post("/api/auth/register", json={
    "email": "major.john@example.com",
    "password": "SecurePass123",
    "full_name": "Major John Smith",
    "service_branch": "Army",
    "rank": "Major",
    "years_of_service": 20
})

# 2. Login and get token
login_response = client.post("/api/auth/login", data={
    "username": "major.john@example.com",
    "password": "SecurePass123"
})
token = login_response.json()["access_token"]

# 3. Update profile with skills
headers = {"Authorization": f"Bearer {token}"}
client.put("/api/profile/me", json={
    "skills": ["Team Leadership", "Strategic Planning", "Risk Management"],
    "location": "New Delhi",
    "preferred_locations": ["Mumbai", "Pune", "Bangalore"]
}, headers=headers)

# 4. Get job recommendations
recommendations = client.get("/api/jobs/recommendations", headers=headers)
print(f"Found {len(recommendations.json())} job recommendations")
```

### 2. Skill Translation Example

```python
# Translate military skills to civilian equivalents
response = client.post("/api/profile/translate-skills", json={
    "military_skills": [
        "Command and Control Operations",
        "Personnel Management",
        "Logistics Coordination",
        "Training and Development"
    ]
}, headers=headers)

civilian_skills = response.json()["translated_skills"]
# Output: ["Operations Management", "Human Resources", "Supply Chain Management", "Training & Development"]
```

## 🚢 Deployment

### Docker Deployment

1. **Build Docker Image**
```bash
docker build -t veterancareer-ai .
```

2. **Run with Docker Compose**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/veterancareer_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=veterancareer_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    
  worker:
    build: .
    command: celery -A app.celery_app worker --loglevel=info
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

### Production Deployment

1. **Environment Setup**
   - Use production database credentials
   - Configure proper JWT secrets
   - Set up SSL certificates
   - Configure reverse proxy (Nginx)

2. **Security Considerations**
   - Change default passwords
   - Use environment-specific secrets
   - Enable rate limiting
   - Configure CORS properly

3. **Monitoring**
   - Set up logging aggregation
   - Monitor database performance
   - Track API response times
   - Monitor background task queue

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make changes and test**
```bash
# Run tests
python -m pytest

# Check code formatting
black app/
flake8 app/
```

4. **Submit pull request**

### Code Style Guidelines

- Follow PEP 8 standards
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features
- Update API documentation

### Adding New Features

1. **Models**: Add new SQLAlchemy models in `app/models/`
2. **Schemas**: Define Pydantic schemas in `app/schemas/`
3. **API Endpoints**: Create routes in `app/api/`
4. **Background Tasks**: Add Celery tasks in `app/tasks/`
5. **ML Features**: Implement in `app/ml/`

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Email: support@veterancareer.ai
- Documentation: https://docs.veterancareer.ai


---

**VeteranCareer AI** - Empowering veterans and government officers in their transition to civilian careers through AI-powered job discovery and career guidance.
