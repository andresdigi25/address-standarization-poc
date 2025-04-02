#!/bin/bash

# Base URL
BASE_URL="http://localhost:8000/api/v1"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Authentication Endpoints ===${NC}\n"

# Register a new user
echo -e "${GREEN}Register a new user:${NC}"
curl -X POST "${BASE_URL}/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpassword123"
  }'

echo -e "\n\n${GREEN}Login to get access token:${NC}"
# Login and get access token
TOKEN=$(curl -X POST "${BASE_URL}/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpassword123" | jq -r '.access_token')

echo -e "\n\n${BLUE}=== Address Management Endpoints ===${NC}\n"

# Create a new address
echo -e "${GREEN}Create a new address:${NC}"
curl -X POST "${BASE_URL}/addresses/" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "postal_code": "10001",
    "country": "USA"
  }'

# Store the created address ID for later use
ADDRESS_ID=$(curl -X POST "${BASE_URL}/addresses/" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "street": "456 Oak Ave",
    "city": "Los Angeles",
    "state": "CA",
    "postal_code": "90001",
    "country": "USA"
  }' | jq -r '.id')

echo -e "\n\n${GREEN}Get all addresses:${NC}"
# Get all addresses
curl -X GET "${BASE_URL}/addresses/" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n${GREEN}Get specific address:${NC}"
# Get specific address
curl -X GET "${BASE_URL}/addresses/${ADDRESS_ID}" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n${GREEN}Update address:${NC}"
# Update address
curl -X PUT "${BASE_URL}/addresses/${ADDRESS_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "street": "456 Oak Avenue",
    "city": "Los Angeles",
    "state": "CA",
    "postal_code": "90001",
    "country": "USA"
  }'

echo -e "\n\n${GREEN}Delete address:${NC}"
# Delete address
curl -X DELETE "${BASE_URL}/addresses/${ADDRESS_ID}" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n${BLUE}=== User Profile Endpoints ===${NC}\n"

# Get current user profile
echo -e "${GREEN}Get current user profile:${NC}"
curl -X GET "${BASE_URL}/auth/me" \
  -H "Authorization: Bearer ${TOKEN}"

echo -e "\n\n${BLUE}=== Health Check Endpoint ===${NC}\n"

# Health check
echo -e "${GREEN}Health check:${NC}"
curl -X GET "http://localhost:8000/health"

echo -e "\n\n${BLUE}=== Root Endpoint ===${NC}\n"

# Root endpoint
echo -e "${GREEN}Root endpoint:${NC}"
curl -X GET "http://localhost:8000/" 