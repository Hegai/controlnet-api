from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ControlNet API"
    
    # Model Settings
    MODEL_PATH: str = "./models/control_sd15_canny.pth"
    MODEL_CONFIG: str = "./models/cldm_v15.yaml"
    
    # Processing Settings
    JOBS_DIR: Path = Path("jobs")
    MAX_IMAGE_SIZE: int = 2048
    
    # GPU Settings
    FORCE_CPU: bool = False
    
    class Config:
        case_sensitive = True

settings = Settings() 