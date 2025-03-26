# Use Ubuntu as base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code and all required directories
COPY app/ ./app/
COPY models/ ./models/
COPY annotator/ ./annotator/
COPY test_imgs/ ./test_imgs/
COPY cldm/ ./cldm/
COPY ldm/ ./ldm/

# Create directory for jobs
RUN mkdir -p /app/jobs

# Set environment variables
ENV PYTHONPATH=/app
ENV FORCE_CPU=0

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]