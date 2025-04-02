FROM python:3.9-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements_xgboot.txt .
RUN pip install --no-cache-dir -r requirements_xgboot.txt

# Copy the Python script
COPY xgboost_address_matching.py .

# Set the entrypoint
ENTRYPOINT ["python", "xgboost_address_matching.py"]