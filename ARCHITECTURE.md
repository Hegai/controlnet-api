# ControlNet API Architecture Documentation

## Overview
The ControlNet API is designed as a modular, scalable service for image generation using ControlNet (https://github.com/lllyasviel/ControlNet.git). This document explains the key architectural decisions and implementation details.

## Core Design Principles

### 1. Modular Architecture
- **Separation of Concerns**: The codebase is organized into distinct modules:
  - `app/api/`: API endpoints and request handling
  - `app/core/`: Core configuration and settings
  - `app/services/`: Business logic and processing
  - `models/`: ML model weights and configurations --> takeover from ControlNet
  - `cldm/` and `ldm/`: Model implementations --> takeover from ControlNet
  - `annotator/`: Image processing utilities

### 2. Asynchronous Processing
- **Job-Based System**:
  - Uses a job-based architecture for handling long-running image generation tasks
  - Each upload creates a unique job ID (UUID)
  - Jobs are processed asynchronously in the background
  - Status tracking and result retrieval endpoints for monitoring progress

### 3. Robust Error Handling
- **Comprehensive Error Management**:
  - Detailed error messages for different failure scenarios
  - Proper logging at each step of processing
  - Graceful handling of file operations and image processing errors
  - Automatic image resizing for large files instead of rejection

### 4. Scalability Considerations
- **Resource Management**:
  - Maximum image size limits (2048px) to prevent memory issues
  - Automatic image resizing while maintaining aspect ratio
  - Efficient file storage using job-based directories
  - Background task processing to handle multiple requests

### 5. Security and Validation
- **Input Validation**:
  - File type validation for uploads
  - Size limits and automatic resizing
  - Job ID verification for all operations
  - Secure file handling and storage

## Implementation Details

### API Endpoints
1. **Upload Endpoint** (`/api/v1/generation/upload/`)
   - Handles image uploads
   - Creates job directory
   - Validates and preprocesses images
   - Returns job ID for tracking

2. **Generation Endpoint** (`/api/v1/generation/{job_id}/generate/`)
   - Triggers image generation process
   - Runs asynchronously
   - Accepts generation parameters
   - Returns immediate job status

3. **Status Endpoint** (`/api/v1/generation/{job_id}/status/`)
   - Checks job progress
   - Returns current status
   - Includes result information when complete

4. **Result Endpoint** (`/api/v1/generation/{job_id}/result/{image_name}`)
   - Retrieves generated images
   - Handles file serving
   - Includes error handling for missing files

### Data Flow
1. **Upload Process**:
   ```
   Client Upload → Validation → Job Creation → File Storage → Job ID Return
   ```

2. **Generation Process**:
   ```
   Job ID → Parameter Validation → Background Processing → Status Updates → Result Storage
   ```

3. **Result Retrieval**:
   ```
   Job ID → Status Check → Result Verification → File Serving → Client Download
   ```

## Technical Decisions

### 1. FastAPI Framework
- Chosen for:
  - High performance and async support
  - Automatic OpenAPI documentation
  - Type safety and validation
  - Easy integration with async operations

### 2. File Storage
- Uses local filesystem with job-based directories
- Each job gets a unique directory
- Structured storage for input and output files
- Easy cleanup and management

### 3. Image Processing
- Automatic resizing for large images
- Maintains aspect ratio
- Uses high-quality LANCZOS resampling
- Preserves image quality while meeting size limits

### 4. Job Management
- UUID-based job identification
- Asynchronous processing
- Status tracking
- Result storage and retrieval
