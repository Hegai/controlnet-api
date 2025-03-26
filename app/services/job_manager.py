from typing import Dict, Optional
import asyncio
from pathlib import Path
import json
import logging

from ..models.generation import JobStatus, GenerationParams
from ..core.config import settings
from .controlnet import ControlNetService

logger = logging.getLogger(__name__)

class JobManager:
    def __init__(self):
        self.controlnet = ControlNetService()
        self._jobs: Dict[str, JobStatus] = {}
        self._lock = asyncio.Lock()

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get the current status of a job."""
        return self._jobs.get(job_id, JobStatus.PENDING)

    async def process_job(self, job_id: str, params: GenerationParams):
        """Process a job asynchronously."""
        async with self._lock:  # Ensure only one job processes at a time
            try:
                self._jobs[job_id] = JobStatus.PROCESSING
                logger.info(f"Starting job {job_id}")
                
                # Process the image
                result_paths = await self.controlnet.process_image(job_id, params)
                
                # Save job metadata
                self._save_job_metadata(job_id, params, result_paths)
                
                self._jobs[job_id] = JobStatus.COMPLETED
                logger.info(f"Completed job {job_id}")
                return result_paths

            except Exception as e:
                logger.error(f"Job {job_id} failed: {str(e)}")
                self._jobs[job_id] = JobStatus.FAILED
                raise

    def _save_job_metadata(
        self,
        job_id: str,
        params: GenerationParams,
        result_paths: list
    ):
        """Save job parameters and results for future reference."""
        job_dir = settings.JOBS_DIR / job_id
        metadata = {
            "parameters": params.model_dump(),
            "results": result_paths,
            "status": JobStatus.COMPLETED
        }
        
        with open(job_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2) 