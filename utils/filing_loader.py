"""Utility to load extracted SEC filing data from S3."""
import json
from typing import Dict, List, Optional
from utils.s3_utils import read_file_from_s3, list_files_in_s3


def load_extracted_filing_for_ticker(ticker: str) -> Optional[Dict]:
    """
    Load the most recent extracted filing JSON for a ticker from data/fillings/{ticker}/.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with extracted sections or None if not found
    """
    try:
        # List files in data/fillings/{ticker}/
        prefix = f"data/fillings/{ticker}/"
        files = list_files_in_s3(prefix)
        
        # Filter for 10-K JSON files
        json_files = [f for f in files if f.endswith('.json') and ('10_k' in f.lower() or '10k' in f.lower())]
        
        if not json_files:
            return None
        
        # Get the most recent file (by filename timestamp if possible)
        # Sort by filename (newest first if timestamp is in name)
        json_files.sort(reverse=True)
        latest_file = json_files[0]
        
        # Read JSON content
        content = read_file_from_s3(latest_file)
        data = json.loads(content)
        
        return data
    except Exception as e:
        print(f"[WARNING] Could not load filing for {ticker}: {e}")
        return None


def load_portfolio_filings(tickers: List[str]) -> Dict[str, Dict]:
    """
    Load extracted filings for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dictionary mapping ticker -> filing data
    """
    filings = {}
    for ticker in tickers:
        filing_data = load_extracted_filing_for_ticker(ticker)
        if filing_data:
            filings[ticker] = filing_data
    return filings


def get_relevant_sections_for_analysis(filing_data: Dict, max_chars_per_section: int = 2000) -> Dict[str, str]:
    """
    Extract and truncate relevant sections for LLM analysis to save tokens.
    
    Args:
        filing_data: Full filing data dictionary
        max_chars_per_section: Maximum characters per section to send to LLM
    
    Returns:
        Dictionary with truncated relevant sections
    """
    if not filing_data or 'sections' not in filing_data:
        return {}
    
    sections = filing_data['sections']
    
    # Focus on high-impact sections for portfolio recommendations
    priority_sections = {
        # High priority for portfolio decisions
        'risk_factors': sections.get('risk_factors', ''),
        'legal_proceedings': sections.get('legal_proceedings', ''),
        'cybersecurity_incidents': sections.get('cybersecurity_incidents', ''),
        'earnings_announcements': sections.get('earnings_announcements', ''),
        'material_impairments': sections.get('material_impairments', ''),
        'acquisitions_dispositions': sections.get('acquisitions_dispositions', ''),
        'accountant_changes': sections.get('accountant_changes', ''),
        
        # Medium priority - financial health
        'mda': sections.get('mda', ''),  # Management Discussion & Analysis
        'market_for_equity': sections.get('market_for_equity', ''),
        'financial_statements': sections.get('financial_statements', ''),
        
        # Context
        'business': sections.get('business', ''),
    }
    
    # Truncate each section to save tokens
    truncated = {}
    for key, value in priority_sections.items():
        if value and len(value) > max_chars_per_section:
            truncated[key] = value[:max_chars_per_section] + "...[truncated]"
        else:
            truncated[key] = value
    
    return truncated

