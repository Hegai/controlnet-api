# ControlNet API

An API for image generation using ControlNet (https://github.com/lllyasviel/ControlNet).

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit
- Download a model from https://huggingface.co/lllyasviel/ControlNet In particular put control_sd15_canny.pth (5.71GB!) in models folder.

## Setup

1. Build the Docker image:
```bash
docker build -t controlnet .
```

2. Run the container:
```bash
docker run --gpus all -p 8000:8000 controlnet
```

## API Endpoints

The API will be available at `http://localhost:8000`

### 1. Upload Image
```bash
curl -X POST -F "file=@test_imgs/house.png" http://localhost:8000/api/v1/generation/upload/
```

### 2. Generate Images
```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "prompt": "house covered in snow",
    "num_samples": 1,
    "image_resolution": 512,
    "ddim_steps": 20,
    "strength": 0.8,
    "scale": 9.0
}' http://localhost:8000/api/v1/generation/{job_id}/generate/
```

### 3. Check Job Status
```bash
curl http://localhost:8000/api/v1/generation/{job_id}/status/
```

### 4. Download Result
```bash
curl -o result.png http://localhost:8000/api/v1/generation/{job_id}/result/result_0.png
```

## Directory Structure

```
ControlNet-minimal/
├── app/                    # FastAPI application
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   └── services/          # Business logic
├── models/                # Model weights and configs
├── cldm/                  # ControlNet model implementation
├── ldm/                   # Latent Diffusion Model
├── annotator/            # Image processing utilities
├── test_imgs/            # Test images
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
└── .dockerignore        # Docker ignore rules
```