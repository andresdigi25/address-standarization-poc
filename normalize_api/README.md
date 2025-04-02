
# Normalize API (FastAPI)

A web service for normalizing input records using field mappings.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open your browser at http://localhost:8000/docs

## 🐳 Run with Docker

```bash
docker build -t normalize-api .
docker run -p 8000:8000 normalize-api
```

## 🔧 Endpoints

- `POST /normalize` - Normalize one record
- `POST /normalize/batch` - Normalize multiple records
- `POST /normalize/upload` - Upload JSON or CSV file
- `GET /logs` - View audit logs


✅ 1. POST /normalize — Normalize a Single Record
Example Payload:

{
  "record": {
    "FName": "Alice",
    "Surname": "Smith",
    "Email_Address": "alice@example.com"
  },
  "mapping_key": "default"
}

http POST http://localhost:8000/normalize record:='{"FName": "Alice", "Surname": "Smith", "Email_Address": "alice@example.com"}'

curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"record": {"FName": "Alice", "Surname": "Smith", "Email_Address": "alice@example.com"}}'


✅ 2. POST /normalize/batch — Normalize Multiple Records
Example Payload:

{
  "records": [
    {
      "FName": "Alice",
      "Surname": "Smith",
      "Email_Address": "alice@example.com"
    },
    {
      "first": "Bob",
      "Last_Name": "Johnson",
      "e-mail": "bob@example.com"
    }
  ],
  "mapping_key": "default"
}

http POST http://localhost:8000/normalize/batch records:='[{"FName": "Alice", "Surname": "Smith", "Email_Address": "alice@example.com"}, {"first": "Bob", "Last_Name": "Johnson", "e-mail": "bob@example.com"}]'

✅ 3. POST /normalize/upload — Upload JSON or CSV File
Using curl (JSON):

curl -X POST http://localhost:8000/normalize/upload \
  -F "file=@sample.json"

Using curl (CSV):

curl -X POST http://localhost:8000/normalize/upload \
  -F "file=@sample.csv"

✅ 4. GET /logs — Get Audit Logs
curl:

curl http://localhost:8000/logs

httpie:

http GET http://localhost:8000/logs