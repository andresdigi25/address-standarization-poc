python facility_cli.py validate-single \
  --atom-id 12345 \
  --source "SOURCE" \
  --name "TEST FACILITY" \
  --addr1 "123 MAIN ST" \
  --city "ANYTOWN" \
  --state "CA" \
  --zip "12345" \
  --auth-type "LICENSE" \
  --auth-id "ABC123" \
  --expire-date "2025-12-31" \
  --first-observed "2023-01-01" \
  --data-type "RETAIL" \
  --class "INDEPENDENT"


python facility_cli.py validate facilities.json

python facility_cli.py validate path/to/your/facilities.json --output valid_facilities.json

python facility_cli.py --help

python facility_cli.py validate --help
python facility_cli.py validate-single --help