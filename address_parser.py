import re

def extract_address_components(address: str) -> dict:
    # Initialize empty components
    components = {
        "street": None,
        "number": None,
        "city": None,
        "state": None,
        "zip_code": None
    }
    
    # Extract ZIP code (5 digits, optionally followed by -4 digits)
    zip_match = re.search(r'(\d{5})(?:-\d{4})?', address)
    if zip_match:
        components["zip_code"] = zip_match.group(0)
        address = address.replace(zip_match.group(0), '')

    # Extract state (2 uppercase letters)
    state_match = re.search(r'\b([A-Z]{2})\b', address)
    if state_match:
        components["state"] = state_match.group(0)
        address = address.replace(state_match.group(0), '')

    # Extract street number
    number_match = re.search(r'^\s*(\d+)', address)
    if number_match:
        components["number"] = number_match.group(1)
        address = address.replace(number_match.group(1), '', 1)

    # Clean remaining address parts
    parts = [p.strip() for p in address.split(',') if p.strip()]
    
    # Assume first part is street name if we have more than one part
    if parts and not components["street"]:
        components["street"] = parts[0].strip()
        
    # Assume last part is city if we have more than one part
    if len(parts) > 1:
        components["city"] = parts[-1].strip()

    return components
