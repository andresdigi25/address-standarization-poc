
import argparse
import csv
import json
from typing import List, Dict, Any
from normalize_record import normalize_record
from colorama import Fore, Style, init
from pathlib import Path

init(autoreset=True)

def load_json_records(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def load_csv_records(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', newline='') as f:
        return list(csv.DictReader(f))

def save_output(file_path: str, data: List[Dict[str, Any]]):
    ext = Path(file_path).suffix.lower()
    with open(file_path, 'w', newline='') as f:
        if ext == ".json":
            json.dump(data, f, indent=2)
        elif ext == ".csv":
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

def print_summary(total_matched, total_skipped, total_unmatched, total_records):
    print(Fore.CYAN + "\n=== Summary Statistics ===")
    print(Fore.CYAN + f"Total Records Processed: {total_records}")
    print(Fore.GREEN + f"Total Fields Mapped:     {total_matched}")
    print(Fore.YELLOW + f"Total Fields Skipped:    {total_skipped}")
    print(Fore.RED + f"Total Fields Unmatched:  {total_unmatched}")

def main():
    parser = argparse.ArgumentParser(description="Normalize messy data records using field mappings.")
    parser.add_argument("input", help="Path to input file (CSV or JSON)")
    parser.add_argument("-o", "--output", help="Optional output file to save normalized results")
    parser.add_argument("-k", "--mapping-key", default="default", help="Field mapping key to use (default: 'default')")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    ext = input_path.suffix.lower()

    if ext == ".json":
        records = load_json_records(args.input)
    elif ext == ".csv":
        records = load_csv_records(args.input)
    else:
        print(Fore.RED + "Unsupported file format. Only .json and .csv are supported.")
        return

    total_matched = 0
    total_skipped = 0
    total_unmatched = 0
    normalized_records = []

    for i, record in enumerate(records, start=1):
        print(Fore.BLUE + f"\n=== Processing Record #{i} ===")
        normalized, audit = normalize_record(record, args.mapping_key)
        normalized_records.append(normalized)

        for entry in audit["matched"]:
            msg = f"{entry['source_field']} → {entry['target_field']} = {entry['value']}"
            if entry["action"] == "mapped":
                print(Fore.GREEN + f"  ✅ {msg}")
                total_matched += 1
            else:
                print(Fore.YELLOW + f"  ⚠️  {msg} (skipped overwrite)")
                total_skipped += 1

        for entry in audit["unmatched"]:
            print(Fore.RED + f"  ❌ {entry['source_field']} = {entry['value']} (no match)")
            total_unmatched += 1

    print_summary(total_matched, total_skipped, total_unmatched, len(records))

    if args.output:
        save_output(args.output, normalized_records)
        print(Fore.BLUE + f"\nSaved normalized output to: {args.output}")

if __name__ == "__main__":
    main()
