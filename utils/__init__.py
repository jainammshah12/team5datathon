"""Utility modules for Regulatory Impact Analyzer."""
from .s3_utils import (
    read_csv_from_s3,
    read_file_from_s3,
    upload_file_to_s3,
    get_sp500_companies,
    get_stock_performance
)
from .document_processor import (
    extract_text_from_html,
    extract_text_from_xml,
    clean_text,
    extract_metadata
)

__all__ = [
    'read_csv_from_s3',
    'read_file_from_s3',
    'upload_file_to_s3',
    'get_sp500_companies',
    'get_stock_performance',
    'extract_text_from_html',
    'extract_text_from_xml',
    'clean_text',
    'extract_metadata'
]

