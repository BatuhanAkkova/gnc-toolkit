# Dockerfile for GNC Toolkit Reproducibility

# Use official Python lightweight image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /usr/src/app

# Install system utilities and build dependencies
# (needed for compiling some C/Fortran modules like pymsis or casadi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy the rest of the source code and tests
COPY . .

# Install the package in editable mode with dev dependencies
RUN pip install -e ".[dev]"

# Ensure entrypoint.sh is executable
RUN chmod +x /usr/src/app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]

# Default command: run tests
CMD ["pytest"]
