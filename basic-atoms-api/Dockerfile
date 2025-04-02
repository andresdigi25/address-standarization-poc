FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ .

# Set environment variables for PostgreSQL
ENV DATABASE_URL=postgresql://user:password@db:5432/atoms_db

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "atom-api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]