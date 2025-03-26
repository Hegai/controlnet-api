from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class GenerationParams(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    a_prompt: str = Field(
        default="good quality",
        description="Additional positive prompt"
    )
    n_prompt: str = Field(
        default="lowres, bad anatomy, worst quality, low quality",
        description="Negative prompt"
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of images to generate"
    )
    image_resolution: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Output image resolution"
    )
    ddim_steps: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of denoising steps"
    )
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Control strength"
    )
    scale: float = Field(
        default=9.0,
        ge=1.0,
        le=20.0,
        description="Guidance scale"
    )
    seed: int = Field(
        default=-1,
        description="Random seed (-1 for random)"
    )
    eta: float = Field(
        default=0.0,
        description="DDIM eta parameter"
    )
    low_threshold: int = Field(
        default=50,
        ge=0,
        le=255,
        description="Canny edge detection low threshold"
    )
    high_threshold: int = Field(
        default=100,
        ge=0,
        le=255,
        description="Canny edge detection high threshold"
    )

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: Optional[str] = None

class GenerationResponse(JobResponse):
    images: Optional[List[str]] = None 