# Atom Data API

A FastAPI application that handles file uploads (CSV and JSON), validates data against the Atom model, maps columns from source files to the model, and stores the data in a PostgreSQL database.

## Features

- Upload CSV and JSON files
- Define and manage mapping profiles for different file formats
- Validate data against the Atom model schema
- Store valid records in a PostgreSQL database
- Track and report validation and parsing errors

## Technology Stack

- FastAPI for API development
- SQLAlchemy v2 and SQLModel for database operations
- PostgreSQL for data storage
- Docker and Docker Compose for containerization
- Pandas for data processing
- Alembic for database migrations

## Project Structure

```
atom-data-api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── models/                  # Database models
│   ├── schemas/                 # Pydantic schemas for request/response
│   ├── services/                # Business logic services
│   └── api/                     # API endpoints
├── migrations/                  # Alembic migrations
├── tests/                       # Unit and integration tests
├── .env                         # Environment variables
├── .gitignore
├── alembic.ini                  # Alembic configuration
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Docker configuration
└── requirements.txt             # Python dependencies
```

## Setup and Installation

### Prerequisites

- Docker and Docker Compose

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd atom-data-api
   ```

2. Start the application:
   ```bash
   docker-compose up -d
   ```

3. The API will be available at http://localhost:8000

4. Access the Swagger documentation at http://localhost:8000/docs

## API Endpoints

### Mapping Profiles

- `GET /api/mappings` - List all mapping profiles
- `GET /api/mappings/{name}` - Get a specific mapping profile
- `POST /api/mappings` - Create a new mapping profile
- `DELETE /api/mappings/{name}` - Delete a mapping profile

### File Upload

- `POST /api/upload` - Upload and process a file

## Example Usage

### Creating a Custom Mapping Profile

```bash
curl -X POST "http://localhost:8000/api/mappings" -H "Content-Type: application/json" -d '{
  "name": "custom_csv",
  "description": "Custom CSV mapping for vendor data",
  "file_type": "csv",
  "columns": [
    {
      "source": "Vendor Name",
      "target": "facility_name",
      "required": true
    },
    {
      "source": "Street Address",
      "target": "addr1",
      "required": true
    },
    {
      "source": "Suite",
      "target": "addr2",
      "required": false,
      "default": ""
    },
    {
      "source": "City",
      "target": "city",
      "required": true
    },
    {
      "source": "State",
      "target": "state",
      "required": true
    },
    {
      "source": "Zip",
      "target": "zip",
      "required": true
    },
    {
      "source": "License Type",
      "target": "auth_type",
      "required": true
    },
    {
      "source": "License Number",
      "target": "auth_id",
      "required": true
    },
    {
      "source": "License Expiration",
      "target": "expire_date",
      "required": false,
      "transformation": "date"
    },
    {
      "source": "First Seen",
      "target": "first_observed",
      "required": false,
      "transformation": "date"
    },
    {
      "source": "Type",
      "target": "data_type",
      "required": true
    },
    {
      "source": "Business Type",
      "target": "class_of_trade",
      "required": true
    }
  ]
}'
```

### Uploading a File

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/data.csv" \
  -F "mapping_name=standard_csv"
```

## Testing

Run tests using pytest:

```bash
docker-compose exec api pytest
```

## Database Migrations

To create a new migration:

```bash
docker-compose exec api alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
docker-compose exec api alembic upgrade head
```

## License

[Your License Information]