# API Endpoints Usage Examples

## Health Check
```bash
curl http://localhost:8000/
```

## Parse Address using Regex
```bash
curl -X POST http://localhost:8000/parse-address \
  -H "Content-Type: application/json" \
  -d '{"address": "123 Main St, Boston, MA 02110"}'
```

## Parse Address using NLTK
```bash
curl -X POST http://localhost:8000/parse-address-nltk \
  -H "Content-Type: application/json" \
  -d '{"address": "123 Main St, Boston, MA 02110"}'
```

## Parse Address using USAddress
```bash
curl -X POST http://localhost:8000/parse-address-usaddress \
  -H "Content-Type: application/json" \
  -d '{"address": "123 Main St, Boston, MA 02110"}'
```

## Compare All Parsers
```bash
curl -X POST http://localhost:8000/compare-parsers \
  -H "Content-Type: application/json" \
  -d '{"address": "123 Main St, Boston, MA 02110"}'
```

Expected Response Format:
```json
{
  "street": "Main St",
  "number": "123",
  "city": "Boston",
  "state": "MA",
  "zip_code": "02110"
}
```

For the compare-parsers endpoint, you'll get results from all parsers:
```json
{
  "regex": { /* address components */ },
  "nltk": { /* address components */ },
  "usaddress": { /* address components */ }
}
```
