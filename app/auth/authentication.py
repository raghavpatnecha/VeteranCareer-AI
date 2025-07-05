from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.models.user import User
from app.auth.jwt_handler import (
    verify_access_token, 
    get_user_from_token, 
    hash_password, 
    verify_password
)
import re

# Security scheme for bearer token
security = HTTPBearer()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password.
    
    Args:
        db: Database session
        email: User email address
        password: Plain text password
    
    Returns:
        Optional[User]: User object if authentication successful, None otherwise
    """
    # Get user by email
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    # Verify password
    if not verify_password(password, user.password_hash):
        return None
    
    # Check if user is active
    if not user.is_active:
        return None
    
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get user by email address.
    
    Args:
        db: Database session
        email: User email address
    
    Returns:
        Optional[User]: User object if found, None otherwise
    """
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """
    Get user by ID.
    
    Args:
        db: Database session
        user_id: User ID
    
    Returns:
        Optional[User]: User object if found, None otherwise
    """
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, user_data: dict) -> User:
    """
    Create new user with hashed password.
    
    Args:
        db: Database session
        user_data: Dictionary containing user information
    
    Returns:
        User: Created user object
    
    Raises:
        HTTPException: If user creation fails
    """
    # Check if user already exists
    existing_user = get_user_by_email(db, user_data["email"])
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Hash password
    hashed_password = hash_password(user_data["password"])
    
    # Create user object
    db_user = User(
        email=user_data["email"],
        password_hash=hashed_password,
        first_name=user_data.get("first_name", ""),
        last_name=user_data.get("last_name", ""),
        phone=user_data.get("phone"),
        service_type=user_data.get("service_type", "military"),
        service_branch=user_data.get("service_branch"),
        rank=user_data.get("rank"),
        years_of_service=user_data.get("years_of_service"),
        current_location=user_data.get("current_location")
    )
    
    # Save to database
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials containing JWT token
        db: Database session
    
    Returns:
        User: Current authenticated user
    
    Raises:
        HTTPException: If authentication fails
    """
    # Extract token from credentials
    token = credentials.credentials
    
    # Get user info from token
    try:
        user_info = get_user_from_token(token)
    except HTTPException:
        raise
    
    # Get user from database
    user = get_user_by_id(db, int(user_info["user_id"]))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to get current active user (additional verification).
    
    Args:
        current_user: Current user from get_current_user dependency
    
    Returns:
        User: Current active user
    
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_current_verified_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to get current verified user.
    
    Args:
        current_user: Current user from get_current_user dependency
    
    Returns:
        User: Current verified user
    
    Raises:
        HTTPException: If user is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    return current_user


def validate_password_strength(password: str) -> bool:
    """
    Validate password strength requirements.
    
    Args:
        password: Password to validate
    
    Returns:
        bool: True if password meets requirements
    
    Raises:
        HTTPException: If password doesn't meet requirements
    """
    # Password requirements
    min_length = 8
    max_length = 128
    
    # Check length
    if len(password) < min_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {min_length} characters long"
        )
    
    if len(password) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be no more than {max_length} characters long"
        )
    
    # Check for at least one uppercase letter
    if not re.search(r"[A-Z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one uppercase letter"
        )
    
    # Check for at least one lowercase letter
    if not re.search(r"[a-z]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one lowercase letter"
        )
    
    # Check for at least one digit
    if not re.search(r"\d", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one digit"
        )
    
    # Check for at least one special character
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one special character"
        )
    
    return True


def validate_email_format(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
    
    Returns:
        bool: True if email format is valid
    
    Raises:
        HTTPException: If email format is invalid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    return True


def validate_phone_format(phone: str) -> bool:
    """
    Validate Indian phone number format.
    
    Args:
        phone: Phone number to validate
    
    Returns:
        bool: True if phone format is valid
    
    Raises:
        HTTPException: If phone format is invalid
    """
    # Indian phone number patterns
    patterns = [
        r'^[6-9]\d{9}$',  # 10 digit mobile number
        r'^\+91[6-9]\d{9}$',  # With +91 country code
        r'^91[6-9]\d{9}$',  # With 91 country code
    ]
    
    phone_valid = any(re.match(pattern, phone) for pattern in patterns)
    
    if not phone_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Indian phone number format"
        )
    
    return True


def update_user_password(db: Session, user: User, new_password: str) -> User:
    """
    Update user password with validation.
    
    Args:
        db: Database session
        user: User object to update
        new_password: New password (plain text)
    
    Returns:
        User: Updated user object
    
    Raises:
        HTTPException: If password update fails
    """
    # Validate new password
    validate_password_strength(new_password)
    
    # Hash new password
    hashed_password = hash_password(new_password)
    
    # Update user password
    try:
        user.password_hash = hashed_password
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update password: {str(e)}"
        )


def verify_user_email(db: Session, user: User) -> User:
    """
    Mark user email as verified.
    
    Args:
        db: Database session
        user: User object to verify
    
    Returns:
        User: Updated user object
    
    Raises:
        HTTPException: If verification fails
    """
    try:
        user.is_verified = True
        user.verification_token = None
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify user: {str(e)}"
        )


def deactivate_user(db: Session, user: User) -> User:
    """
    Deactivate user account.
    
    Args:
        db: Database session
        user: User object to deactivate
    
    Returns:
        User: Updated user object
    
    Raises:
        HTTPException: If deactivation fails
    """
    try:
        user.is_active = False
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate user: {str(e)}"
        )


def reactivate_user(db: Session, user: User) -> User:
    """
    Reactivate user account.
    
    Args:
        db: Database session
        user: User object to reactivate
    
    Returns:
        User: Updated user object
    
    Raises:
        HTTPException: If reactivation fails
    """
    try:
        user.is_active = True
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reactivate user: {str(e)}"
        )


# Optional dependency that doesn't raise exception if user is not authenticated
def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optional dependency to get current user without raising exception if not authenticated.
    
    Args:
        credentials: Optional HTTP authorization credentials
        db: Database session
    
    Returns:
        Optional[User]: Current user if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        user_info = get_user_from_token(token)
        user = get_user_by_id(db, int(user_info["user_id"]))
        
        if user and user.is_active:
            return user
    except:
        pass
    
    return None
