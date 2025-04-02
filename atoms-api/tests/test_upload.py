import pytest
import io
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_mappings(client):
    response = client.get("/api/mappings")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "standard_csv" in response.json()
    assert "standard_json" in response.json()


def test_upload_csv_with_standard_mapping(client):
    # Create a simple CSV with standard headers
    csv_content = """Facility,Address1,Address2,City,State,ZIP,Authorization Type,Authorization ID,Expiration Date,First Observed,Data Type,Class of Trade
Test Facility,123 Main St,Suite 100,Anytown,TX,12345,License,ABC123,2023-12-31,2022-01-01,Sample,Healthcare
"""
    
    # Create a file-like object
    file = io.BytesIO(csv_content.encode())
    file.name = "test.csv"
    
    # Upload the file
    response = client.post(
        "/api/upload",
        files={"file": ("test.csv", file, "text/csv")},
        data={"mapping_name": "standard_csv"}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["successful_records"] == 1
    assert response.json()["failed_records"] == 0


def test_upload_invalid_csv(client):
    # Create a CSV with missing required fields
    csv_content = """Facility,Address1,City
Test Facility,123 Main St,Anytown
"""
    
    # Create a file-like object
    file = io.BytesIO(csv_content.encode())
    file.name = "test.csv"
    
    # Upload the file
    response = client.post(
        "/api/upload",
        files={"file": ("test.csv", file, "text/csv")},
        data={"mapping_name": "standard_csv"}
    )
    
    # Check response
    assert response.status_code == 400
    assert "Required columns missing" in response.text