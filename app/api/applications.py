from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.models.application import Application, ApplicationStatus, FollowUpType, FollowUpReminder
from app.models.job import Job
from app.models.user import User
from app.schemas.application import (
    ApplicationCreateSchema, 
    ApplicationUpdateSchema, 
    ApplicationResponseSchema,
    FollowUpReminderSchema
)
from app.auth.authentication import get_current_user
# from app.services.notification_service import send_notification # Optional, if notifications are implemented

router = APIRouter()

@router.post("/", response_model=ApplicationResponseSchema, status_code=status.HTTP_201_CREATED)
async def create_application(
    application_data: ApplicationCreateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new job application record.
    """
    try:
        job = db.query(Job).filter(Job.id == application_data.job_id).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found."
            )

        existing_application = db.query(Application).filter(
            Application.user_id == current_user.id,
            Application.job_id == application_data.job_id
        ).first()
        if existing_application:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Application for this job already exists for the current user."
            )

        new_application = Application(
            user_id=current_user.id,
            job_id=application_data.job_id,
            status=ApplicationStatus.DRAFT,
            applied_date=datetime.now() if application_data.status == ApplicationStatus.APPLIED else None,
            **application_data.dict(exclude_unset=True, exclude={"job_id", "user_id"})
        )
        
        if application_data.status == ApplicationStatus.APPLIED:
            new_application.applied_date = datetime.now()
            new_application.status = ApplicationStatus.APPLIED
        elif application_data.status:
            new_application.status = application_data.status

        db.add(new_application)
        db.commit()
        db.refresh(new_application)
        return ApplicationResponseSchema(**new_application.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create application: {str(e)}"
        )

@router.get("/", response_model=List[ApplicationResponseSchema])
async def get_applications(
    status: Optional[ApplicationStatus] = Query(None, description="Filter applications by status"),
    job_title: Optional[str] = Query(None, description="Filter applications by job title substring"),
    company_name: Optional[str] = Query(None, description="Filter applications by company name substring"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve a list of job applications for the current user.
    Supports filtering by status, job title, and company name.
    """
    try:
        query = db.query(Application).filter(Application.user_id == current_user.id)

        if status:
            query = query.filter(Application.status == status)
        
        if job_title:
            query = query.join(Job).filter(Job.title.ilike(f"%{job_title}%"))

        if company_name:
            query = query.join(Job).filter(Job.company_name.ilike(f"%{company_name}%"))

        applications = query.offset(offset).limit(limit).all()
        return [ApplicationResponseSchema(**app.to_dict()) for app in applications]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve applications: {str(e)}"
        )

@router.get("/{application_id}", response_model=ApplicationResponseSchema)
async def get_application_details(
    application_id: int = Path(..., description="The ID of the application to retrieve"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve details of a specific job application by its ID.
    """
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()

    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you do not have permission to access it."
        )
    return ApplicationResponseSchema(**application.to_dict())

@router.put("/{application_id}", response_model=ApplicationResponseSchema)
async def update_application(
    application_id: int,
    application_data: ApplicationUpdateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing job application record.
    """
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()

    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you do not have permission to update it."
        )

    try:
        update_data = application_data.dict(exclude_unset=True)
        
        if "status" in update_data and update_data["status"] != application.status:
            notes_for_status_update = update_data.get("user_notes", "") 
            application.update_status(update_data["status"], notes=notes_for_status_update)
            del update_data["status"]
            if "user_notes" in update_data:
                del update_data["user_notes"]

        for field, value in update_data.items():
            setattr(application, field, value)

        db.commit()
        db.refresh(application)
        return ApplicationResponseSchema(**application.to_dict())

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update application: {str(e)}"
        )

@router.delete("/{application_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_application(
    application_id: int = Path(..., description="The ID of the application to delete"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a job application record.
    """
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()

    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you do not have permission to delete it."
        )

    try:
        db.delete(application)
        db.commit()
        return {}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete application: {str(e)}"
        )

@router.post("/{application_id}/follow-up", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_follow_up_reminder(
    application_id: int,
    reminder_data: FollowUpReminderSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a follow-up reminder for a specific job application.
    """
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()

    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you do not have permission to access it."
        )

    try:
        new_reminder = FollowUpReminder(
            application_id=application_id,
            user_id=current_user.id,
            reminder_type=reminder_data.reminder_type,
            reminder_date=reminder_data.reminder_date,
            title=reminder_data.title,
            description=reminder_data.description,
            email_notification=True
        )
        
        db.add(new_reminder)
        db.commit()
        db.refresh(new_reminder)

        # if new_reminder.email_notification:
        #     send_notification(current_user.email, new_reminder.title, new_reminder.description, new_reminder.reminder_date)

        return new_reminder.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create follow-up reminder: {str(e)}"
        )

@router.get("/{application_id}/follow-up", response_model=List[FollowUpReminderSchema])
async def get_application_follow_ups(
    application_id: int = Path(..., description="Application ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve all follow-up reminders for a specific job application.
    """
    application = db.query(Application).filter(
        Application.id == application_id,
        Application.user_id == current_user.id
    ).first()

    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found or you do not have permission to access it."
        )

    try:
        follow_ups = application.follow_ups
        return [FollowUpReminderSchema(**fu.to_dict()) for fu in follow_ups]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve follow-up reminders: {str(e)}"
        )

@router.put("/follow-up/{reminder_id}", response_model=FollowUpReminderSchema)
async def update_follow_up_reminder(
    reminder_id: int,
    reminder_data: FollowUpReminderSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing follow-up reminder.
    """
    reminder = db.query(FollowUpReminder).filter(
        FollowUpReminder.id == reminder_id,
        FollowUpReminder.user_id == current_user.id
    ).first()

    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Follow-up reminder not found or you do not have permission to update it."
        )
    
    try:
        update_data = reminder_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(reminder, field, value)
        
        db.commit()
        db.refresh(reminder)
        return FollowUpReminderSchema(**reminder.to_dict())
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update follow-up reminder: {str(e)}"
        )

@router.delete("/follow-up/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_follow_up_reminder(
    reminder_id: int = Path(..., description="Reminder ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a follow-up reminder.
    """
    reminder = db.query(FollowUpReminder).filter(
        FollowUpReminder.id == reminder_id,
        FollowUpReminder.user_id == current_user.id
    ).first()

    if not reminder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Follow-up reminder not found or you do not have permission to delete it."
        )
    
    try:
        db.delete(reminder)
        db.commit()
        return {}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete follow-up reminder: {str(e)}"
        )