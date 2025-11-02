"""Utility functions for S3 operations."""

import boto3
import pandas as pd
import os
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize S3 client
bucket_name = os.getenv("S3_BUCKET_NAME")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")

try:
    # Get AWS credentials from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Create S3 client with explicit credentials
    client_config = {
        "service_name": "s3",
        "region_name": aws_region,
    }

    # Use explicit credentials if available (more reliable)
    if aws_access_key and aws_secret_key:
        client_config["aws_access_key_id"] = aws_access_key
        client_config["aws_secret_access_key"] = aws_secret_key

    # Add session token if present (required for temporary credentials like AWS SSO)
    if aws_session_token:
        client_config["aws_session_token"] = aws_session_token
        print(f"[INFO] Using temporary AWS credentials (with session token)")

    s3_client = boto3.client(**client_config)

    if bucket_name:
        print(f"[INFO] S3 client initialized for bucket: {bucket_name}")
    else:
        print(f"[WARNING] S3 client initialized but S3_BUCKET_NAME not set")
except Exception as e:
    s3_client = None
    print(f"[ERROR] S3 client not initialized: {e}")
    print("[ERROR] Please configure AWS credentials in .env file")
    print("[ERROR] Run 'python test_env_credentials.py' to diagnose issues")


def read_csv_from_s3(key: str) -> pd.DataFrame:
    """Read a CSV file from S3 and return as pandas DataFrame."""
    if s3_client is None:
        raise ConnectionError(
            "S3 not configured. Please check AWS credentials in .env file."
        )

    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    except Exception as e:
        raise ConnectionError(
            f"Failed to read CSV '{key}' from S3 bucket '{bucket_name}': {e}"
        )


def read_file_from_s3(key: str) -> str:
    """Read a text file from S3 and return as string."""
    if s3_client is None:
        raise ConnectionError(
            "S3 not configured. Please check AWS credentials in .env file."
        )

    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception as e:
        raise ConnectionError(
            f"Failed to read '{key}' from S3 bucket '{bucket_name}': {e}"
        )


def check_file_exists_in_s3(key: str) -> bool:
    """Check if a file exists in S3."""
    if s3_client is None:
        raise ConnectionError(
            "S3 not configured. Please check AWS credentials in .env file."
        )

    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        print(f"[INFO] File exists in S3: {key}")
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"[INFO] File does not exist in S3: {key}")
            return False
        # Re-raise other errors
        raise
    except Exception as e:
        print(f"[WARNING] Could not check file existence: {e}")
        return False


def delete_file_from_s3(key: str) -> None:
    """Delete a file from S3."""
    if s3_client is None:
        raise ConnectionError(
            "S3 not configured. Please check AWS credentials in .env file."
        )

    try:
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        print(f"[INFO] Deleted file from S3: {key}")
    except Exception as e:
        raise ConnectionError(
            f"Failed to delete '{key}' from S3 bucket '{bucket_name}': {e}"
        )


def upload_file_to_s3(file_content: bytes, key: str, overwrite: bool = True) -> None:
    """
    Upload a file to S3.

    Args:
        file_content: File content as bytes
        key: S3 key (path) for the file
        overwrite: If True, delete existing file before uploading
    """
    if s3_client is None:
        raise ConnectionError(
            "S3 not configured. Please check AWS credentials in .env file."
        )

    # Delete existing file if overwrite is enabled
    if overwrite and check_file_exists_in_s3(key):
        print(f"[INFO] Overwriting existing file: {key}")
        delete_file_from_s3(key)

    s3_client.put_object(Bucket=bucket_name, Key=key, Body=file_content)


def list_files_in_s3(prefix: str) -> list:
    """List all files in S3 with the given prefix."""
    if s3_client is None:
        raise ConnectionError(
            "S3 not configured. Please check AWS credentials in .env file."
        )

    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" in response:
            files = [obj["Key"] for obj in response["Contents"]]
            print(f"[INFO] Found {len(files)} files in S3 bucket with prefix: {prefix}")
            return files
        print(f"[INFO] No files found in S3 with prefix: {prefix}")
        return []
    except Exception as e:
        raise ConnectionError(f"Failed to list S3 files with prefix '{prefix}': {e}")


def get_sp500_companies() -> pd.DataFrame:
    """Get S&P 500 companies list from S3."""
    return read_csv_from_s3("data/2025-08-15_composition_sp500.csv")


def get_stock_performance() -> pd.DataFrame:
    """Get stock performance data from S3."""
    return read_csv_from_s3("data/2025-09-26_stocks-performance.csv")


def get_available_directives() -> list:
    """Get list of available regulatory documents from S3."""
    return list_files_in_s3("data/directives/")


def get_raw_filings(ticker: str = None) -> list:
    """Get list of raw SEC filing HTML/XML files from S3 data/fillings/."""
    if ticker:
        return list_files_in_s3(f"data/fillings/{ticker}/")
    return list_files_in_s3("data/fillings/")


def get_available_filings(ticker: str = None) -> list:
    """Get list of available extracted company filings from S3."""
    if ticker:
        return list_files_in_s3(f"extracted_filings/{ticker}/")
    return list_files_in_s3("extracted_filings/")
