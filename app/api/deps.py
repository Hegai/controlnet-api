from fastapi import HTTPException, Path
import os
from ..core.config import settings

async def verify_job_id(
    job_id: str = Path(..., description="The ID of the generation job")
) -> str:
    """
    Dependency to verify that a job exists and return its ID.
    """
    job_path = settings.JOBS_DIR / job_id
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return job_id 