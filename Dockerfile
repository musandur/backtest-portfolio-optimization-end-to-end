# Use the official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project into the container
COPY . /app/

# Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose Flask port
EXPOSE 5001

# # Run the app
# CMD ["python", "app.py"]

# Run with Gunicorn
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
# CMD ["gunicorn", "--workers=1", "--threads=2", "--bind=0.0.0.0:5001", "app:app"]
CMD ["gunicorn", "--worker-class=gevent", "--bind=0.0.0.0:5001", "--workers=1", "app:app"]

