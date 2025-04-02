# Address Standardization API

A FastAPI-based REST API for address standardization and validation. This API provides endpoints for managing addresses with proper validation and standardization.

## Features

- User authentication with JWT tokens
- Address CRUD operations
- Address validation and standardization
- PostgreSQL database integration
- Docker support
- Comprehensive API documentation
- Health check endpoints

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- PostgreSQL (if running locally)
- Make (optional, for using Makefile commands)

## Project Structure

```
address-api/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   └── addresses.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── models/
│   │   ├── user.py
│   │   └── address.py
│   ├── services/
│   │   ├── user_service.py
│   │   └── address_service.py
│   └── atom-api.py
├── tests/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
└── api_curl_commands.sh
```

### Import Structure

The project uses relative imports to maintain clean and maintainable code structure. Here's how the import syntax works:

- `.` - Current directory
- `..` - Parent directory
- `...` - Parent's parent directory

Examples:
```python
# In src/api/routes/auth.py
from ...core.config import get_settings  # Goes up 3 levels to find core/config.py
from ..dependencies import get_current_user  # Goes up 2 levels to find dependencies.py

# In src/services/address_service.py
from ..models.address import Address  # Goes up 2 levels to find models/address.py
from ..core.exceptions import AddressNotFoundError  # Goes up 2 levels to find core/exceptions.py
```

This import structure helps maintain the modularity of the code and makes it easier to refactor or move files without breaking imports.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd address-api
```

### 2. Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Update the `.env` file with your configuration:
```env
# API Settings
API_V1_STR=/api/v1
PROJECT_NAME=Address Standardization API

# Database Settings
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_SERVER=db
POSTGRES_PORT=5432
POSTGRES_DB=atoms_db

# Security Settings
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging Settings
LOG_LEVEL=INFO
```

### 3. Running with Docker (Recommended)

1. Build and start the containers:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

### 4. Running Locally (Alternative)

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API:
```bash
uvicorn src.atom-api:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once the API is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing the API

A comprehensive set of curl commands is provided in `api_curl_commands.sh`. To use it:

1. Make the script executable:
```bash
chmod +x api_curl_commands.sh
```

2. Run the script:
```bash
./api_curl_commands.sh
```

## Development Tools

The project includes a Makefile with common development tasks:

```bash
# Build Docker images
make build

# Start services
make up

# Stop services
make down

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Clean up
make clean
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register a new user
- `POST /api/v1/auth/login` - Login to get access token
- `GET /api/v1/auth/me` - Get current user profile

### Addresses
- `POST /api/v1/addresses/` - Create a new address
- `GET /api/v1/addresses/` - Get all addresses
- `GET /api/v1/addresses/{address_id}` - Get specific address
- `PUT /api/v1/addresses/{address_id}` - Update an address
- `DELETE /api/v1/addresses/{address_id}` - Delete an address

### System
- `GET /health` - Health check endpoint
- `GET /` - Root endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.