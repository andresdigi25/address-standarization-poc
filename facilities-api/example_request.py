import requests

url = "http://localhost:8000/facilities/"
payload = {
    "facility_id": 0,
    "source": "example_source",
    "facility_name": "Example Facility",
    "addr1": "123 Example St",
    "addr2": "Suite 100",
    "city": "Example City",
    "state": "EX",
    "zip": "12345",
    "auth_type": "example_auth",
    "auth_id": "auth123",
    "expire_date": "2025-04-02",
    "first_observed": "2025-04-02",
    "data_type": "example_data",
    "class_of_trade": "example_trade"
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
