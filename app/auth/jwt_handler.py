from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from app.config import settings
import uuid

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings from config
SECRET_KEY = settings.jwt_secret_key
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = settings.refresh_token_expire_days


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token with user data.
    
    Args:
        data: Dictionary containing user information to encode in token
        expires_delta: Optional custom expiration time
    
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Add standard JWT claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),  # JWT ID for token tracking
        "type": "access"
    })
    
    # Encode and return token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create JWT refresh token with user data.
    
    Args:
        data: Dictionary containing user information to encode in token
    
    Returns:
        str: Encoded refresh JWT token
    """
    to_encode = data.copy()
    
    # Set expiration time for refresh token (longer duration)
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Add standard JWT claims
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),
        "type": "refresh"
    })
    
    # Encode and return token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string to verify
    
    Returns:
        Dict[str, Any]: Decoded token payload
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Validate required fields
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user identification",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_access_token(token: str) -> Dict[str, Any]:
    """
    Verify access token specifically.
    
    Args:
        token: JWT access token string to verify
    
    Returns:
        Dict[str, Any]: Decoded token payload
    
    Raises:
        HTTPException: If token is invalid, expired, or not an access token
    """
    payload = verify_token(token)
    
    # Check if token is access token
    token_type = payload.get("type")
    if token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type for access",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


def verify_refresh_token(token: str) -> Dict[str, Any]:
    """
    Verify refresh token specifically.
    
    Args:
        token: JWT refresh token string to verify
    
    Returns:
        Dict[str, Any]: Decoded token payload
    
    Raises:
        HTTPException: If token is invalid, expired, or not a refresh token
    """
    payload = verify_token(token)
    
    # Check if token is refresh token
    token_type = payload.get("type")
    if token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type for refresh",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


def get_user_from_token(token: str) -> Dict[str, Any]:
    """
    Extract user information from JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Dict[str, Any]: User information from token
    
    Raises:
        HTTPException: If token is invalid or user info is missing
    """
    payload = verify_access_token(token)
    
    # Extract user information
    user_info = {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "user_type": payload.get("user_type", "user"),
        "is_verified": payload.get("is_verified", False),
        "token_id": payload.get("jti"),
        "issued_at": payload.get("iat"),
        "expires_at": payload.get("exp")
    }
    
    # Validate required user info
    if not user_info["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user identification",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_info


def create_token_pair(user_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Create both access and refresh tokens for a user.
    
    Args:
        user_data: User information to encode in tokens
    
    Returns:
        Dict[str, str]: Dictionary containing access_token and refresh_token
    """
    # Prepare token data
    token_data = {
        "sub": str(user_data.get("user_id")),  # Subject (user ID)
        "email": user_data.get("email"),
        "user_type": user_data.get("user_type", "user"),
        "is_verified": user_data.get("is_verified", False),
        "service_branch": user_data.get("service_branch"),
        "rank": user_data.get("rank")
    }
    
    # Create tokens
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


def refresh_access_token(refresh_token: str) -> Dict[str, str]:
    """
    Generate new access token using refresh token.
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        Dict[str, str]: New token pair
    
    Raises:
        HTTPException: If refresh token is invalid
    """
    # Verify refresh token
    payload = verify_refresh_token(refresh_token)
    
    # Extract user data for new token
    user_data = {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "user_type": payload.get("user_type", "user"),
        "is_verified": payload.get("is_verified", False),
        "service_branch": payload.get("service_branch"),
        "rank": payload.get("rank")
    }
    
    # Create new token pair
    return create_token_pair(user_data)


def is_token_expired(token: str) -> bool:
    """
    Check if token is expired without raising exception.
    
    Args:
        token: JWT token string
    
    Returns:
        bool: True if token is expired, False otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        if exp is None:
            return True
        
        return datetime.utcfromtimestamp(exp) < datetime.utcnow()
    except JWTError:
        return True


def get_token_remaining_time(token: str) -> Optional[timedelta]:
    """
    Get remaining time before token expires.
    
    Args:
        token: JWT token string
    
    Returns:
        Optional[timedelta]: Remaining time or None if token is invalid/expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        if exp is None:
            return None
        
        exp_datetime = datetime.utcfromtimestamp(exp)
        remaining = exp_datetime - datetime.utcnow()
        
        return remaining if remaining.total_seconds() > 0 else None
    except JWTError:
        return None


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password to verify against
    
    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_password_reset_token(email: str) -> str:
    """
    Create password reset token.
    
    Args:
        email: User email address
    
    Returns:
        str: Password reset token
    """
    data = {
        "sub": email,
        "type": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    }
    
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify password reset token and extract email.
    
    Args:
        token: Password reset token
    
    Returns:
        Optional[str]: Email address if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != "password_reset":
            return None
        
        return payload.get("sub")
    except JWTError:
        return None
