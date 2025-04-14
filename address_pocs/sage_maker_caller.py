import boto3
import json

def lambda_handler(event, context):
    # Get request body from API Gateway
    request_body = json.loads(event['body'])
    
    # Prepare input for SageMaker
    addresses_to_match = request_body['addresses']
    
    # Call SageMaker endpoint
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='address-matching-endpoint',
        ContentType='application/json',
        Body=json.dumps(addresses_to_match)
    )
    
    # Process response
    result = json.loads(response['Body'].read().decode())
    
    # Return API response
    return {
        'statusCode': 200,
        'body': json.dumps(result),
        'headers': {
            'Content-Type': 'application/json'
        }
    }