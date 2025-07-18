# official slim Python base
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# System-level dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY app/ .

# Create upload directory
RUN mkdir -p /app/uploads

# Expose FastAPI port
EXPOSE 8000

# Set environment
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
