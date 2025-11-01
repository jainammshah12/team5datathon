"""Utility functions for S3 operations."""
import boto3
import pandas as pd
import os
from io import StringIO, BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize S3 client
bucket_name = os.getenv('S3_BUCKET_NAME')

# Try to initialize S3 client, but don't crash if credentials are missing
try:
    s3_client = boto3.client('s3')
except Exception as e:
    s3_client = None
    print(f"Warning: S3 client not initialized. AWS credentials may be missing: {e}")

def read_csv_from_s3(key: str) -> pd.DataFrame:
    """Read a CSV file from S3 and return as pandas DataFrame."""
    if s3_client is None:
        raise ConnectionError("S3 not configured. Please check AWS credentials in .env file.")
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    except Exception as e:
        # Fallback to local file if available
        local_path = key.replace('data/', './data/')
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        raise ConnectionError(f"Failed to read from S3 and no local file found: {e}")

def read_file_from_s3(key: str) -> str:
    """Read a text file from S3 and return as string."""
    if s3_client is None:
        raise ConnectionError("S3 not configured. Please check AWS credentials in .env file.")
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return obj['Body'].read().decode('utf-8')
    except Exception as e:
        # Fallback to local file if available
        local_path = key.replace('data/', './data/')
        if os.path.exists(local_path):
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()
        raise ConnectionError(f"Failed to read from S3 and no local file found: {e}")

def upload_file_to_s3(file_content: bytes, key: str) -> None:
    """Upload a file to S3."""
    if s3_client is None:
        raise ConnectionError("S3 not configured. Please check AWS credentials in .env file.")
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=file_content)

def list_files_in_s3(prefix: str) -> list:
    """List all files in S3 with the given prefix."""
    if s3_client is None:
        # Fallback to local directory listing
        local_path = prefix.replace('data/', './data/')
        if os.path.exists(local_path) and os.path.isdir(local_path):
            files = []
            for root, dirs, filenames in os.walk(local_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename).replace('\\', '/')
                    s3_key = file_path.replace('./data/', 'data/')
                    files.append(s3_key)
            return files
        return []
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

def get_sp500_companies() -> pd.DataFrame:
    """Get S&P 500 companies list from S3."""
    return read_csv_from_s3("data/2025-08-15_composition_sp500.csv")

def get_stock_performance() -> pd.DataFrame:
    """Get stock performance data from S3."""
    return read_csv_from_s3("data/2025-09-26_stocks-performance.csv")

def get_available_directives() -> list:
    """Get list of available regulatory documents from S3."""
    return list_files_in_s3("data/directives/")

def get_available_filings(ticker: str = None) -> list:
    """Get list of available company filings from S3."""
    if ticker:
        return list_files_in_s3(f"data/fillings/{ticker}/")
    return list_files_in_s3("data/fillings/")

