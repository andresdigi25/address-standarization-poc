.PHONY: build up down clean logs ps shell test lint

# Build the Docker images
build:
	docker compose build

# Start the application
up:
	docker compose up

# Start the application in detached mode
up-d:
	docker compose up -d

# Stop the application
down:
	docker compose down

# Stop the application and remove volumes
down-v:
	docker compose down -v

# Clean up Docker resources
clean:
	docker compose down -v --rmi all --remove-orphans

# View logs
logs:
	docker compose logs -f

# List running containers
ps:
	docker compose ps

# Open a shell in the API container
shell:
	docker compose exec api /bin/bash

# Run tests (if you add them later)
test:
	docker compose exec api python -m pytest

# Run linting with ruff
lint:
	docker compose exec api ruff check .

# Format code with ruff
format:
	docker compose exec api ruff format .

# Create data directory if it doesn't exist
init:
	mkdir -p data

# Help command
help:
	@echo "Available commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make up        - Start the application"
	@echo "  make up-d      - Start the application in detached mode"
	@echo "  make down      - Stop the application"
	@echo "  make down-v    - Stop the application and remove volumes"
	@echo "  make clean     - Clean up all Docker resources"
	@echo "  make logs      - View application logs"
	@echo "  make ps        - List running containers"
	@echo "  make shell     - Open a shell in the API container"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run linting with ruff"
	@echo "  make format    - Format code with ruff"
	@echo "  make init      - Create data directory"
	@echo "  make help      - Show this help message" 