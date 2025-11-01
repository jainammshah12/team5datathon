import boto3, pandas as pd, os
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

s3 = boto3.client('s3')
bucket = os.getenv('S3_BUCKET_NAME')

def read_csv_from_s3(key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

sp500 = read_csv_from_s3("data/2025-08-15_composition_sp500.csv")
stocks = read_csv_from_s3("data/2025-09-26_stocks-performance.csv")
