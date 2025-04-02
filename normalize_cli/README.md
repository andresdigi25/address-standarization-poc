
# Normalize CLI Tool

A simple command-line tool to normalize messy input records using a configurable field mapping.

## 📦 Requirements

- Python 3.7+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Normalize a JSON file:

```bash
python cli.py sample.json -o output.json
```

### Normalize a CSV file:

```bash
python cli.py sample.csv -o output.csv
```

### Optional arguments:

- `-k <mapping_key>` to use a specific field mapping (default is `'default'`)

## 🐳 Run with Docker

### Build the Docker image:

```bash
docker build -t normalize-cli .
```

### Run normalization:

```bash
docker run --rm -v $(pwd):/app normalize-cli python cli.py /app/sample.json -o /app/output.json
```

Replace `sample.json` or `sample.csv` with your input file.
