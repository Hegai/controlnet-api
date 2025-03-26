from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import uuid
from PIL import Image
import io
import logging
from pathlib import Path

from ...models.generation import (
    GenerationParams,
    JobResponse,
    GenerationResponse,
    JobStatus
)
from ...core.config import settings
from ...services.job_manager import JobManager
from ..deps import verify_job_id

logger = logging.getLogger(__name__)
router = APIRouter()
job_manager = JobManager()

@router.post("/upload/", response_model=JobResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file to process")
) -> JobResponse:
    """Upload an image for processing."""
    try:
        # Log the upload attempt
        logger.info(f"Received upload request for file: {file.filename}")

        # Validate image
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file")

        try:
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Successfully opened image with size: {image.size}")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )

        # Resize image if too large while maintaining aspect ratio
        original_size = image.size
        new_size = original_size
        if max(image.size) > settings.MAX_IMAGE_SIZE:
            ratio = settings.MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {original_size} to {new_size}")

        # Create job
        job_id = str(uuid.uuid4())
        job_dir = settings.JOBS_DIR / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created job directory: {job_dir}")
        except Exception as e:
            logger.error(f"Failed to create job directory: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create job directory: {str(e)}"
            )

        # Save image
        image_path = job_dir / "input.png"
        try:
            image.save(image_path)
            logger.info(f"Saved image to: {image_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save image: {str(e)}"
            )

        logger.info(f"Created new job: {job_id}")
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Image uploaded successfully" + (f" and resized from {original_size} to {new_size}" if new_size != original_size else "")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during upload: {str(e)}"
        )

@router.post("/{job_id}/generate/", response_model=JobResponse)
async def generate_images(
    params: GenerationParams,
    background_tasks: BackgroundTasks,
    job_id: str = Depends(verify_job_id)
) -> JobResponse:
    """Start image generation for a job."""
    try:
        # Start processing in background
        background_tasks.add_task(
            job_manager.process_job,
            job_id,
            params
        )

        logger.info(f"Started generation for job: {job_id}")
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            message="Generation started"
        )
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/status/", response_model=GenerationResponse)
async def get_job_status(
    job_id: str = Depends(verify_job_id)
) -> GenerationResponse:
    """Get the current status of a job."""
    try:
        status = job_manager.get_job_status(job_id)

        # If job is completed, return results
        if status == JobStatus.COMPLETED:
            job_dir = settings.JOBS_DIR / job_id
            result_files = [f.name for f in job_dir.glob("result_*.png")]
            return GenerationResponse(
                job_id=job_id,
                status=status,
                images=result_files
            )

        return GenerationResponse(
            job_id=job_id,
            status=status
        )

    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/result/{image_name}")
async def get_result_image(
    image_name: str,
    job_id: str = Depends(verify_job_id)
):
    """Get a generated image by name."""
    try:
        image_path = settings.JOBS_DIR / job_id / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Image retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))