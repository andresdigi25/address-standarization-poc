.PHONY: build up down test lint format clean

# Build the Docker images
build:
	docker-compose build

# Start the services
up:
	docker-compose up

# Stop the services
down:
	docker-compose down

# Run tests
test:
	pytest tests/

# Run linting
lint:
	flake8 src/ tests/
	mypy src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} + 