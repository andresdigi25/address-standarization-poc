import usaddress


def extract_address_usaddress(address: str) -> dict:
    try:
        tagged_address, address_type = usaddress.tag(address)
        
        components = {
            "street": None,
            "number": None,
            "city": None,
            "state": None,
            "zip_code": None
        }
        
        # Map usaddress tags to our format
        if 'AddressNumber' in tagged_address:
            components['number'] = tagged_address['AddressNumber']
        
        if 'StreetName' in tagged_address:
            street_parts = []
            if 'StreetNamePreDirectional' in tagged_address:
                street_parts.append(tagged_address['StreetNamePreDirectional'])
            street_parts.append(tagged_address['StreetName'])
            if 'StreetNamePostType' in tagged_address:
                street_parts.append(tagged_address['StreetNamePostType'])
            components['street'] = ' '.join(street_parts)
            
        if 'PlaceName' in tagged_address:
            components['city'] = tagged_address['PlaceName']
            
        if 'StateName' in tagged_address:
            components['state'] = tagged_address['StateName']
            
        if 'ZipCode' in tagged_address:
            components['zip_code'] = tagged_address['ZipCode']
            
        return components
    except Exception as e:
        raise ValueError(f"Error parsing address with usaddress: {str(e)}")

