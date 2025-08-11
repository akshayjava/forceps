# Use an official NVIDIA CUDA runtime image as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python3 and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set the working directory
WORKDIR /app

# Copy the application code
# We copy requirements and install scripts first to leverage Docker layer caching
COPY app/requirements.txt app/requirements.txt
COPY app/install.sh app/install.sh

# Create a virtual environment and install dependencies
# The install.sh script handles this
RUN chmod +x app/install.sh && ./app/install.sh

# Copy the rest of the application code
COPY . .

# Make scripts executable
RUN chmod +x app/enqueue_jobs.py app/engine.py app/build_index.py

# The entrypoint will be specified in docker-compose.yml
# This makes the image flexible for running different services.
# Set a default command to keep the container running if needed.
CMD ["/bin/bash"]
