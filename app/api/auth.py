from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from app.database import get_db
from app.models.user import User
from app.schemas.user import (
    UserRegistrationSchema,
    UserLoginSchema,
    UserResponseSchema,
    UserProfileResponseSchema,
    UserProfileUpdateSchema,
    PasswordChangeSchema,
    PasswordResetRequestSchema,
    PasswordResetSchema,
    AuthTokenSchema,
    TokenRefreshSchema,
    EmailVerificationSchema,
    UserStatsSchema,
    ServiceBranchesResponseSchema,
    MILITARY_RANKS,
    GOVERNMENT_SERVICES,
    PARAMILITARY_FORCES
)
from app.auth.authentication import (
    authenticate_user,
    create_user,
    get_current_user,
    get_current_active_user,
    get_current_verified_user,
    get_user_by_email,
    validate_password_strength,
    validate_email_format,
    update_user_password,
    verify_user_email
)
from app.auth.jwt_handler import (
    create_token_pair,
    refresh_access_token,
    verify_password_reset_token,
    create_password_reset_token,
    is_token_expired
)

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring.
    
    Returns service status without requiring authentication.
    """
    return {"status": "healthy", "service": "VeteranCareer AI"}


@router.post("/register", response_model=AuthTokenSchema, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegistrationSchema,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    Creates a new user account with the provided information and returns
    authentication tokens for immediate login.
    """
    try:
        # Validate email format
        validate_email_format(user_data.email)
        
        # Validate password strength
        validate_password_strength(user_data.password)
        
        # Check if user already exists
        existing_user = get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email address already exists"
            )
        
        # Prepare user data for creation
        user_create_data = {
            "email": user_data.email,
            "password": user_data.password,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "phone": user_data.phone,
            "service_type": user_data.service_type,
            "service_branch": user_data.service_branch,
            "rank": user_data.rank,
            "years_of_service": user_data.years_of_service,
            "current_location": user_data.current_location
        }
        
        # Create user
        new_user = create_user(db, user_create_data)
        
        # Calculate initial profile completion score
        new_user.calculate_profile_completion()
        db.commit()
        
        # Create authentication tokens
        token_data = {
            "user_id": new_user.id,
            "email": new_user.email,
            "user_type": "user",
            "is_verified": new_user.is_verified,
            "service_branch": new_user.service_branch,
            "rank": new_user.rank
        }
        
        tokens = create_token_pair(token_data)
        
        # Update last login
        new_user.last_login = datetime.now()
        db.commit()
        
        return AuthTokenSchema(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            user=UserResponseSchema(**new_user.to_dict())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=AuthTokenSchema)
async def login_user(
    user_credentials: UserLoginSchema,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access tokens.
    
    Validates user credentials and returns JWT tokens for authentication.
    """
    try:
        # Authenticate user
        user = authenticate_user(db, user_credentials.email, user_credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create authentication tokens
        token_data = {
            "user_id": user.id,
            "email": user.email,
            "user_type": "user",
            "is_verified": user.is_verified,
            "service_branch": user.service_branch,
            "rank": user.rank
        }
        
        tokens = create_token_pair(token_data)
        
        # Update last login timestamp
        user.last_login = datetime.now()
        db.commit()
        
        return AuthTokenSchema(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            user=UserResponseSchema(**user.to_dict())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/refresh", response_model=Dict[str, str])
async def refresh_token(
    token_data: TokenRefreshSchema
):
    """
    Refresh access token using refresh token.
    
    Generates a new access token using a valid refresh token.
    """
    try:
        # Refresh tokens
        new_tokens = refresh_access_token(token_data.refresh_token)
        
        return {
            "access_token": new_tokens["access_token"],
            "refresh_token": new_tokens["refresh_token"],
            "token_type": new_tokens["token_type"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """
    Logout current user.
    
    Note: In a stateless JWT implementation, logout is typically handled
    client-side by removing tokens. This endpoint is provided for
    consistency and future token blacklisting implementation.
    """
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserProfileResponseSchema)
async def get_current_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's detailed profile information.
    """
    # Update profile completion score
    current_user.calculate_profile_completion()
    db.commit()
    
    return UserProfileResponseSchema(**current_user.to_dict())


@router.put("/me", response_model=UserProfileResponseSchema)
async def update_user_profile(
    profile_data: UserProfileUpdateSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update current user's profile information.
    """
    try:
        # Update only provided fields
        update_data = profile_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(current_user, field) and value is not None:
                setattr(current_user, field, value)
        
        # Update profile completion score
        current_user.calculate_profile_completion()
        
        # Save changes
        db.commit()
        db.refresh(current_user)
        
        return UserProfileResponseSchema(**current_user.to_dict())
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Change user's password.
    """
    try:
        # Verify current password
        if not current_user.check_password(password_data.current_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        update_user_password(db, current_user, password_data.new_password)
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


@router.post("/forgot-password")
async def request_password_reset(
    reset_request: PasswordResetRequestSchema,
    db: Session = Depends(get_db)
):
    """
    Request password reset for user account.
    
    Sends password reset token to user's email address.
    """
    try:
        # Check if user exists
        user = get_user_by_email(db, reset_request.email)
        
        # Don't reveal whether user exists or not for security
        if user and user.is_active:
            # Create password reset token
            reset_token = create_password_reset_token(user.email)
            
            # TODO: Send email with reset token
            # This would integrate with email service
            # For now, we'll just return success message
            pass
        
        return {
            "message": "If an account with this email exists, a password reset link has been sent"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset request failed: {str(e)}"
        )


@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordResetSchema,
    db: Session = Depends(get_db)
):
    """
    Reset user password using reset token.
    """
    try:
        # Verify reset token
        email = verify_password_reset_token(reset_data.token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Get user by email
        user = get_user_by_email(db, email)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        # Update password
        update_user_password(db, user, reset_data.new_password)
        
        return {"message": "Password reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset failed: {str(e)}"
        )


@router.post("/verify-email")
async def verify_email(
    verification_data: EmailVerificationSchema,
    db: Session = Depends(get_db)
):
    """
    Verify user's email address using verification token.
    """
    try:
        # TODO: Implement email verification token validation
        # This would verify the token and mark user as verified
        
        return {"message": "Email verification endpoint - implementation pending"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email verification failed: {str(e)}"
        )


@router.get("/profile/stats", response_model=UserStatsSchema)
async def get_user_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user's application and profile statistics.
    """
    try:
        # TODO: Implement actual statistics calculation
        # This would query applications, matches, etc.
        
        stats = UserStatsSchema(
            total_applications=0,
            active_applications=0,
            interviews_scheduled=0,
            offers_received=0,
            profile_views=0,
            job_matches_count=0,
            average_match_score=0.0
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )


@router.get("/service-branches", response_model=ServiceBranchesResponseSchema)
async def get_service_branches():
    """
    Get available service branches for registration/profile update.
    """
    return ServiceBranchesResponseSchema(
        military=MILITARY_RANKS,
        government=GOVERNMENT_SERVICES,
        paramilitary=PARAMILITARY_FORCES
    )


@router.get("/ranks/{service_branch}")
async def get_ranks_by_service_branch(service_branch: str):
    """
    Get available ranks for a specific service branch.
    """
    if service_branch in MILITARY_RANKS:
        return {
            "service_branch": service_branch,
            "ranks": MILITARY_RANKS[service_branch]
        }
    elif service_branch in GOVERNMENT_SERVICES:
        return {
            "service_branch": service_branch,
            "ranks": []  # Government services don't have standardized ranks
        }
    elif service_branch in PARAMILITARY_FORCES:
        return {
            "service_branch": service_branch,
            "ranks": []  # Paramilitary forces have varied rank structures
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service branch not found"
        )


@router.delete("/account")
async def delete_user_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete current user's account.
    
    This deactivates the account rather than permanently deleting it.
    """
    try:
        # Deactivate account instead of permanent deletion
        current_user.is_active = False
        db.commit()
        
        return {"message": "Account deactivated successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Account deletion failed: {str(e)}"
        )


@router.post("/reactivate")
async def reactivate_account(
    user_credentials: UserLoginSchema,
    db: Session = Depends(get_db)
):
    """
    Reactivate a deactivated user account.
    """
    try:
        # Get user by email
        user = get_user_by_email(db, user_credentials.email)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User account not found"
            )
        
        # Verify password
        if not user.check_password(user_credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password"
            )
        
        # Reactivate account
        user.is_active = True
        user.last_login = datetime.now()
        db.commit()
        
        # Create tokens for immediate login
        token_data = {
            "user_id": user.id,
            "email": user.email,
            "user_type": "user",
            "is_verified": user.is_verified,
            "service_branch": user.service_branch,
            "rank": user.rank
        }
        
        tokens = create_token_pair(token_data)
        
        return AuthTokenSchema(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            user=UserResponseSchema(**user.to_dict())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Account reactivation failed: {str(e)}"
        )