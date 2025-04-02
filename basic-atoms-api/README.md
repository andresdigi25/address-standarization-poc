# Address Standardization POC

This project is a FastAPI application for managing atoms with a PostgreSQL database. It provides a RESTful API for creating, reading, updating, and deleting atom records.

## Project Structure

```
address-standarization-poc
├── src
│   ├── atom-api.py        # FastAPI application code
│   └── __init__.py       # Package initialization
├── Dockerfile             # Dockerfile for building the application image
├── docker-compose.yml     # Docker Compose configuration for services
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd address-standarization-poc
   ```

2. **Build and run the application using Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the API:**
   The FastAPI application will be available at `http://localhost:8000`. You can access the interactive API documentation at `http://localhost:8000/docs`.

## Usage

- **Create an Atom:**
  - Endpoint: `POST /atoms/`
  - Request Body: JSON representation of the Atom model.

- **Read All Atoms:**
  - Endpoint: `GET /atoms/`

- **Read a Single Atom:**
  - Endpoint: `GET /atoms/{atom_id}`

- **Update an Atom:**
  - Endpoint: `PUT /atoms/{atom_id}`
  - Request Body: JSON representation of the updated Atom model.

- **Delete an Atom:**
  - Endpoint: `DELETE /atoms/{atom_id}`

## Database Configuration

The application uses PostgreSQL as the database. Ensure that the database service is running and accessible as defined in the `docker-compose.yml` file.

## Dependencies

- FastAPI
- SQLModel
- PostgreSQL

## License

This project is licensed under the MIT License.