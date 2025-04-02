
from fastapi import FastAPI, UploadFile, File
from app.schemas import RecordInput, BatchInput
from app.normalize_record import normalize_record, read_audit_logs
import csv
import io
import json

app = FastAPI()

@app.post("/normalize")
def normalize_single(data: RecordInput):
    result, audit = normalize_record(data.record, data.mapping_key)
    return {"normalized": result, "audit": audit}

@app.post("/normalize/batch")
def normalize_batch(data: BatchInput):
    results = []
    audits = []
    for record in data.records:
        result, audit = normalize_record(record, data.mapping_key)
        results.append(result)
        audits.append(audit)
    return {"normalized": results, "audits": audits}

@app.post("/normalize/upload")
async def normalize_file(file: UploadFile = File(...), mapping_key: str = "default"):
    ext = file.filename.lower()
    contents = await file.read()

    if ext.endswith(".json"):
        records = json.loads(contents)
    elif ext.endswith(".csv"):
        decoded = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))
        records = list(reader)
    else:
        return {"error": "Unsupported file format"}

    results, audits = [], []
    for record in records:
        result, audit = normalize_record(record, mapping_key)
        results.append(result)
        audits.append(audit)

    return {"normalized": results, "audits": audits}

@app.get("/logs")
def get_logs():
    return {"logs": read_audit_logs()}
