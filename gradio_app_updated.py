"""Main Gradio application for Regulatory Impact Analyzer."""
import gradio as gr
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import os

# Import utility modules
from utils.s3_utils import (
    get_sp500_companies,
    get_stock_performance,
    get_available_directives,
    get_available_filings,
    read_file_from_s3,
    list_files_in_s3
)
from utils.document_processor import (
    extract_text_from_html,
    extract_text_from_xml,
    clean_text,
    extract_metadata
)
from utils.portfolio_manager import (
    parse_portfolio_csv,
    parse_portfolio_manual,
    validate_portfolio,
    portfolio_to_dataframe,
    get_portfolio_summary
)
from utils.portfolio_storage import (
    load_portfolio_from_s3,
    save_portfolio_to_s3,
    list_portfolios_in_s3,
    add_portfolio_entry,
    calculate_portfolio_value,
    get_portfolio_key
)
from utils.s3_utils import get_available_filings
from utils.sec_filing_extractor import extract_key_filing_sections
from utils.yfinance_fetcher import fetch_daily_stock_data
from utils.filing_loader import load_portfolio_filings, get_relevant_sections_for_analysis
from llm.llm_client import get_llm_client

# Initialize LLM client
llm_client = get_llm_client()

# Global cache for data
_sp500_data = None
_stock_performance = None

def load_sp500_data() -> pd.DataFrame:
    """Load S&P 500 companies data (cached)."""
    global _sp500_data
    if _sp500_data is None:
        try:
            _sp500_data = get_sp500_companies()
        except Exception as e:
            # Return error DataFrame
            return pd.DataFrame({
                "Error": [f"⚠️ Failed to load S&P 500 data: {str(e)}"],
                "Solution": ["Please check AWS credentials in .env file or ensure local data/ folder exists"]
            })
    return _sp500_data

def load_stock_performance() -> pd.DataFrame:
    """Load stock performance data (cached)."""
    global _stock_performance
    if _stock_performance is None:
        try:
            _stock_performance = get_stock_performance()
        except Exception as e:
            # Return error DataFrame
            return pd.DataFrame({
                "Error": [f"⚠️ Failed to load stock performance data: {str(e)}"],
                "Solution": ["Please check AWS credentials in .env file or ensure local data/ folder exists"]
            })
    return _stock_performance

def get_directive_list() -> List[str]:
    """Get list of available directives from S3."""
    try:
        directives = get_available_directives()
        # Filter to show only actual files (not directories)
        return [d for d in directives if '.' in d.split('/')[-1] and 'README' not in d]
    except Exception as e:
        return [f"Error loading directives: {str(e)}"]

def load_directives_for_recommendations() -> Dict[str, Dict[str, str]]:
    """
    Load regulatory directives from S3 for use in portfolio recommendations.
    Processes raw HTML/XML files from data/directives/ directory.
    
    Returns:
        Dictionary mapping directive name -> extracted sections
    """
    directives_data = {}
    try:
        # Get available directives from S3 (data/directives/)
        directive_files = get_available_directives()
        
        # Filter for HTML/XML files (raw directives)
        raw_files = [d for d in directive_files if d.endswith(('.html', '.xml', '.htm'))]
        
        # Also check for JSON files (pre-extracted)
        json_files = [d for d in directive_files if d.endswith('.json')]
        
        # First try to load pre-extracted JSON files
        for directive_path in json_files[:3]:  # Limit to 3 JSON files
            try:
                content = read_file_from_s3(directive_path)
                if content:
                    import json
                    directive_data = json.loads(content)
                    directive_name = directive_path.split('/')[-1].replace('.json', '').replace('extracted_', '')
                    directives_data[directive_name] = directive_data
                    print(f"[INFO] Loaded pre-extracted directive: {directive_name}")
            except Exception as e:
                print(f"[WARNING] Could not load JSON directive {directive_path}: {e}")
                continue
        
        # Process raw HTML/XML files if we don't have enough
        if len(directives_data) < 3 and raw_files:
            try:
                from utils.directive_analyzer import (
                    extract_sections_from_directive, 
                    is_xml_content, 
                    detect_language,
                    extract_full_text_from_html
                )
                directive_analyzer_available = True
            except ImportError:
                print("[WARNING] directive_analyzer not available, skipping raw directive processing")
                directive_analyzer_available = False
            
            if directive_analyzer_available:
                # Process up to 3 raw directives
                for directive_path in raw_files[:3]:
                    if len(directives_data) >= 5:  # Total limit of 5 directives
                        break
                    
                    try:
                        print(f"[INFO] Processing raw directive: {directive_path}")
                        content = read_file_from_s3(directive_path)
                        if not content:
                            continue
                        
                        # Detect format and language
                        is_xml = is_xml_content(content)
                        text = extract_full_text_from_html(content, is_xml=is_xml)
                        
                        # Detect language (fallback to 'en' if detection fails)
                        try:
                            language, _ = detect_language(text)
                        except:
                            language = 'en'
                        
                        # Extract sections
                        sections = extract_sections_from_directive(content, is_xml=is_xml, language=language)
                        
                        # Extract directive name from path
                        directive_name = directive_path.split('/')[-1].replace('.html', '').replace('.xml', '').replace('.htm', '')
                        directives_data[directive_name] = sections
                        print(f"[INFO] Extracted and loaded directive: {directive_name}")
                        
                    except Exception as e:
                        print(f"[WARNING] Could not process raw directive {directive_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
    except Exception as e:
        print(f"[WARNING] Error loading directives: {e}")
        import traceback
        traceback.print_exc()
    
    return directives_data


def load_document(document_path: str) -> Tuple[str, str]:
    """Load and process a document from S3."""
    try:
        content = read_file_from_s3(document_path)
        metadata = extract_metadata(document_path)
        
        # Extract text based on file type
        if document_path.endswith('.html'):
            text = extract_text_from_html(content)
        elif document_path.endswith('.xml'):
            text = extract_text_from_xml(content)
        else:
            text = content
        
        text = clean_text(text)
        
        # Format metadata
        metadata_str = f"**File:** {metadata['filename']}\n"
        if metadata['date']:
            metadata_str += f"**Date:** {metadata['date']}\n"
        if metadata['type']:
            metadata_str += f"**Type:** {metadata['type']}\n"
        if metadata['ticker']:
            metadata_str += f"**Ticker:** {metadata['ticker']}\n"
        
        preview = text[:1000] + "..." if len(text) > 1000 else text
        
        return text, metadata_str + f"\n**Preview:**\n{preview}"
    except Exception as e:
        return "", f"Error loading document: {str(e)}"

def analyze_document(document_text: str) -> str:
    """Analyze document and extract entities."""
    if not document_text:
        return "Please load a document first."
    
    try:
        # Extract entities using LLM
        entities = llm_client.extract_entities(document_text)
        
        # Format output
        result = "## Extracted Entities\n\n"
        result += f"**Document Type:** {entities.get('document_type', 'Unknown')}\n"
        result += f"**Jurisdiction:** {entities.get('jurisdiction', 'Unknown')}\n"
        result += f"**Effective Date:** {entities.get('effective_date', 'Unknown')}\n\n"
        
        if entities.get('key_requirements'):
            result += "**Key Requirements:**\n"
            for req in entities['key_requirements']:
                result += f"- {req}\n"
            result += "\n"
        
        if entities.get('affected_sectors'):
            result += "**Affected Sectors:**\n"
            for sector in entities['affected_sectors']:
                result += f"- {sector}\n"
        
        return result
    except Exception as e:
        return f"Error analyzing document: {str(e)}"

def evaluate_impact(entities_json: str) -> str:
    """Evaluate financial impact on S&P 500 companies."""
    try:
        entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
        sp500_data = load_sp500_data()
        
        # Convert DataFrame to dict for LLM
        companies_dict = sp500_data.to_dict('records')[:50]  # Limit for demo
        
        # Analyze impact
        impact = llm_client.analyze_impact(entities, companies_dict)
        
        # Format output
        result = "## Impact Analysis\n\n"
        result += f"**Most Affected Sector:** {impact['sector_summary'].get('most_affected_sector', 'Unknown')}\n"
        result += f"**Overall Impact:** {impact['sector_summary'].get('overall_impact', 'Pending analysis')}\n\n"
        
        if impact.get('affected_companies'):
            result += "**Affected Companies:**\n\n"
            for company in impact['affected_companies'][:10]:  # Show top 10
                result += f"**{company.get('ticker', 'N/A')}** - {company.get('company_name', 'N/A')}\n"
                result += f"  Impact: {company.get('impact_level', 'Unknown')} ({company.get('impact_type', 'Unknown')})\n"
                result += f"  Rationale: {company.get('rationale', 'N/A')}\n\n"
        
        return result
    except Exception as e:
        return f"Error evaluating impact: {str(e)}"

def fetch_price_for_date(ticker: str, date: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Fetch historical stock price for a given date.
    
    Args:
        ticker: Stock symbol
        date: Date in YYYY-MM-DD format
    
    Returns:
        Tuple of (price, error_message)
    """
    try:
        # Fetch data for the date (use a small range around the date)
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        start_date = (date_obj - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = (date_obj + timedelta(days=5)).strftime('%Y-%m-%d')
        
        # Fetch data
        data = fetch_daily_stock_data(ticker, start_date, end_date)
        
        if data.empty:
            return None, f"No data found for {ticker} around {date}"
        
        # Find closest date
        # Convert Date column to datetime if needed
        if data['Date'].dtype == 'object':
            data['Date'] = pd.to_datetime(data['Date'])
        target_date = pd.to_datetime(date)
        
        # Get closest date
        data['date_diff'] = abs(pd.to_datetime(data['Date']) - target_date)
        closest_row = data.loc[data['date_diff'].idxmin()]
        
        price = float(closest_row['Close'])
        
        return price, None
        
    except Exception as e:
        return None, f"Error fetching price: {str(e)}"


def add_stock_to_portfolio(
    ticker: str,
    quantity: int,
    date_bought: str,
    price: float,
    current_portfolio_df: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """
    Add a stock entry to the portfolio.
    Either date_bought OR price must be provided (not both required).
    If date is provided, fetches historical price automatically.
    
    Returns:
        Tuple of (status_message, updated_dataframe)
    """
    # FIXED: Better input validation
    if not ticker or not quantity:
        return "⚠️ Please provide ticker and quantity", current_portfolio_df
    
    # Validate that either date or price is provided (but not necessarily both)
    has_date = date_bought and date_bought.strip()
    has_price = price and price > 0
    
    if not has_date and not has_price:
        return "⚠️ Please provide either purchase date OR price", current_portfolio_df
    
    try:
        # Validate inputs
        ticker = ticker.upper().strip()
        quantity = int(quantity)
        
        if quantity <= 0:
            return "❌ Quantity must be greater than 0", current_portfolio_df
        
        # FIXED: Validate date format if provided
        if has_date:
            from datetime import datetime
            try:
                date_obj = datetime.strptime(date_bought.strip(), '%Y-%m-%d')
                if date_obj > datetime.now():
                    return "❌ Date cannot be in the future", current_portfolio_df
            except ValueError:
                return "❌ Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-15)", current_portfolio_df
        
        # Fetch price if date is provided
        final_price = None
        if has_date:
            fetched_price, error = fetch_price_for_date(ticker, date_bought)
            if error:
                return f"❌ {error}", current_portfolio_df
            final_price = fetched_price
            final_date = date_bought
        else:
            # Use provided price
            final_price = float(price)
            # Use today's date if no date provided
            from datetime import datetime
            final_date = datetime.now().strftime('%Y-%m-%d')
        
        if final_price <= 0:
            return "❌ Price must be greater than 0", current_portfolio_df
        
        # Add entry
        updated_df = add_portfolio_entry(ticker, final_price, quantity, final_date, current_portfolio_df)
        
        if has_date:
            return f"✅ Added {quantity} shares of {ticker} @ ${final_price:.2f} (price on {final_date})", updated_df
        else:
            return f"✅ Added {quantity} shares of {ticker} @ ${final_price:.2f}", updated_df
        
    except ValueError as e:
        return f"❌ Invalid input: {str(e)}", current_portfolio_df
    except Exception as e:
        return f"❌ Error: {str(e)}", current_portfolio_df


def save_portfolio_to_s3_handler(portfolio_df: pd.DataFrame, portfolio_name: str = None) -> str:
    """
    Save current portfolio to S3 with optional name.
    If name not provided or name exists, auto-generates unique name.
    
    Returns:
        Status message
    """
    if portfolio_df is None or len(portfolio_df) == 0:
        return "⚠️ Portfolio is empty. Add some stocks first."
    
    try:
        # Generate filename if not provided
        if not portfolio_name or not portfolio_name.strip():
            portfolio_name = "portfolio"
        
        # Clean portfolio name (remove invalid characters)
        portfolio_name = "".join(c for c in portfolio_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not portfolio_name:
            portfolio_name = "portfolio"
        
        # Check for existing portfolios and find unique name
        existing_portfolios = list_portfolios_in_s3()
        base_filename = f"{portfolio_name}.csv"
        filename = base_filename
        
        # If filename exists, append number
        if base_filename in existing_portfolios:
            counter = 1
            while filename in existing_portfolios:
                filename = f"{portfolio_name}_{counter}.csv"
                counter += 1
            if counter > 1:
                status_msg = f"⚠️ Portfolio name '{portfolio_name}' already exists. Saved as '{filename}'\n\n"
            else:
                status_msg = ""
        else:
            status_msg = ""
        
        success, error = save_portfolio_to_s3(portfolio_df, filename=filename)
        
        if success:
            return status_msg + f"✅ Portfolio '{filename}' saved to S3 successfully! ({len(portfolio_df)} holdings)"
        else:
            return f"❌ Failed to save portfolio: {error}"
    except Exception as e:
        return f"❌ Error saving to S3: {str(e)}"


def load_portfolio_from_s3_handler(filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Load portfolio from S3 and extract SEC filing data for each ticker.
    
    Returns:
        Tuple of (DataFrame, status_message)
    """
    if not filename or filename == "Select a portfolio...":
        return pd.DataFrame(), ""
    
    try:
        df, error = load_portfolio_from_s3(filename)
        
        if df is not None and len(df) > 0:
            status_msg = f"✅ Loaded portfolio '{filename}' with {len(df)} holdings\n\n"
            status_msg += "📄 **SEC Filing Extraction:**\n"
            
            # Extract portfolio name from filename (e.g., "portfolio_name.csv" -> "portfolio_name")
            portfolio_name = filename.replace('.csv', '').replace('portfolio_', '')
            
            # Extract SEC filing data for each unique ticker
            print(f"[INFO] Starting SEC filing extraction for portfolio companies...")
            unique_tickers = df['Ticker'].unique()
            
            extraction_count = 0
            skipped_count = 0
            
            for ticker in unique_tickers:
                try:
                    # First, check if extracted data already exists in S3
                    # Check in data/fillings/{ticker}/ for extracted JSON files
                    from utils.s3_utils import list_files_in_s3
                    extracted_prefix = f"data/fillings/{ticker}/"
                    existing_extractions = list_files_in_s3(extracted_prefix)
                    
                    # Filter for 10-K JSON files (should have "10_K" or "10k" in name)
                    existing_10k_files = [f for f in existing_extractions if ('10_k' in f.lower() or '10k' in f.lower()) and f.endswith('.json')]
                    
                    if existing_10k_files:
                        # Data already exists, skip extraction
                        print(f"[INFO] Extracted data already exists for {ticker}, skipping extraction")
                        status_msg += f"✓ {ticker}: Using existing extraction\n"
                        skipped_count += 1
                        continue
                    
                    # No existing data, proceed with extraction
                    filings = get_available_filings(ticker)
                    
                    if filings:
                        # Find 10-K filing
                        filing_10k = [f for f in filings if '10k' in f.lower()]
                        
                        if filing_10k:
                            filing_path = filing_10k[0]
                            print(f"[INFO] Found 10-K filing for {ticker}: {filing_path}")
                            
                            try:
                                # Read the HTML content from S3
                                from utils.s3_utils import read_file_from_s3
                                html_content = read_file_from_s3(filing_path)
                                
                                if html_content:
                                    # Extract sections using sec-parser and save to S3
                                    sections = extract_key_filing_sections(
                                        ticker=ticker,
                                        html_content=html_content,
                                        portfolio_name=portfolio_name,
                                        filing_type="10-K",
                                        save_to_s3=True
                                    )
                                    
                                    if sections.get('_extraction_success'):
                                        status_msg += f"✅ {ticker}: Extracted and saved\n"
                                        extraction_count += 1
                                    else:
                                        error_msg = sections.get('_error', 'Unknown error')
                                        status_msg += f"⚠️ {ticker}: Extraction failed - {error_msg}\n"
                                else:
                                    status_msg += f"⚠️ {ticker}: Could not read filing from S3\n"
                                    
                            except Exception as e:
                                print(f"[ERROR] Failed to extract {ticker}: {e}")
                                status_msg += f"❌ {ticker}: {str(e)}\n"
                        else:
                            print(f"[WARNING] No 10-K filing found for {ticker}")
                            status_msg += f"⚠️ {ticker}: No 10-K filing\n"
                    else:
                        print(f"[INFO] No filings in S3 for {ticker}")
                        status_msg += f"ℹ️ {ticker}: No filings in S3\n"
                        
                except Exception as e:
                    print(f"[WARNING] Could not process {ticker}: {e}")
                    status_msg += f"❌ {ticker}: {str(e)}\n"
            
            status_msg += f"\n💾 Extracted data location: `data/fillings/`\n"
            status_msg += f"📊 Status: {extraction_count} extracted, {skipped_count} using existing data ({extraction_count + skipped_count}/{len(unique_tickers)} ready)"
            
            return df, status_msg
        else:
            return pd.DataFrame(), f"⚠️ Portfolio '{filename}' is empty or not found"
    except Exception as e:
        return pd.DataFrame(), f"❌ Error loading portfolio: {str(e)}"


def upload_portfolio_csv(file) -> Tuple[str, pd.DataFrame, str]:
    """
    Handle CSV file upload for portfolio.
    
    Returns:
        Tuple of (status_message, portfolio_df, portfolio_json)
    """
    if file is None:
        return "❌ No file uploaded", pd.DataFrame(), ""
    
    try:
        # Handle both filepath string and file object
        file_path = file if isinstance(file, str) else file.name
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Parse portfolio
        portfolio, error = parse_portfolio_csv(file_content)
        
        if error:
            return f"❌ {error}", pd.DataFrame(), ""
        
        # Validate
        is_valid, validation_error = validate_portfolio(portfolio)
        if not is_valid:
            return f"❌ Validation error: {validation_error}", pd.DataFrame(), ""
        
        # Convert to DataFrame
        df = portfolio_to_dataframe(portfolio)
        portfolio_json = json.dumps(portfolio)
        
        return f"✅ Portfolio uploaded successfully! {len(portfolio['holdings'])} holdings loaded.", df, portfolio_json
        
    except Exception as e:
        return f"❌ Error uploading portfolio: {str(e)}", pd.DataFrame(), ""


def upload_portfolio_manual(portfolio_text: str) -> Tuple[str, pd.DataFrame, str]:
    """
    Handle manual text input for portfolio.
    
    Returns:
        Tuple of (status_message, portfolio_df, portfolio_json)
    """
    if not portfolio_text or not portfolio_text.strip():
        return "⚠️ Please enter portfolio holdings", pd.DataFrame(), ""
    
    try:
        # Parse portfolio
        portfolio, error = parse_portfolio_manual(portfolio_text)
        
        if error:
            return f"❌ {error}", pd.DataFrame(), ""
        
        # Validate
        is_valid, validation_error = validate_portfolio(portfolio)
        if not is_valid:
            return f"❌ Validation error: {validation_error}", pd.DataFrame(), ""
        
        # Convert to DataFrame
        df = portfolio_to_dataframe(portfolio)
        portfolio_json = json.dumps(portfolio)
        
        return f"✅ Portfolio loaded successfully! {len(portfolio['holdings'])} holdings.", df, portfolio_json
        
    except Exception as e:
        return f"❌ Error loading portfolio: {str(e)}", pd.DataFrame(), ""


def generate_portfolio_recommendations_from_filings(portfolio_df: pd.DataFrame) -> str:
    """
    Generate portfolio recommendations based on SEC filing data.
    Uses extracted filing sections to provide actionable recommendations.
    
    Args:
        portfolio_df: DataFrame with portfolio holdings (Ticker, Price, Quantity, Date_Bought)
    
    Returns:
        Formatted markdown string with recommendations
    """
    # FIXED: Handle None and empty DataFrames
    if portfolio_df is None:
        portfolio_df = pd.DataFrame()
    
    if len(portfolio_df) == 0:
        return (
            "⚠️ **Portfolio is empty.**\n\n"
            "Please add stocks to your portfolio first using:\n"
            "1. Load from S3 dropdown, OR\n"
            "2. Manually add stocks using the 'Add Stock' form"
        )
    
    try:
        # Get portfolio tickers and calculate weights
        tickers = portfolio_df['Ticker'].unique().tolist()
        total_cost = (portfolio_df['Price'] * portfolio_df['Quantity']).sum()
        
        portfolio_weights = {}
        portfolio_purchase_info = {}  # Store purchase price, quantity, date for each ticker
        for ticker in tickers:
            ticker_rows = portfolio_df[portfolio_df['Ticker'] == ticker]
            ticker_cost = (ticker_rows['Price'] * ticker_rows['Quantity']).sum()
            portfolio_weights[ticker] = ticker_cost / total_cost if total_cost > 0 else 0
            
            # Store purchase info for performance analysis
            portfolio_purchase_info[ticker] = {
                'avg_purchase_price': ticker_cost / ticker_rows['Quantity'].sum() if ticker_rows['Quantity'].sum() > 0 else 0,
                'total_quantity': ticker_rows['Quantity'].sum(),
                'total_cost': ticker_cost,
                'earliest_date': ticker_rows['Date_Bought'].min() if 'Date_Bought' in ticker_rows.columns else None
            }
        
        # Fetch current prices for performance analysis
        print(f"[INFO] Fetching current prices for {len(tickers)} holdings...")
        current_prices = {}
        for ticker in tickers:
            try:
                from utils.yfinance_fetcher import fetch_stock_info
                info = fetch_stock_info(ticker)
                current_price = info.get('Current Price')
                if current_price:
                    current_prices[ticker] = current_price
                    print(f"[INFO] {ticker}: Current price ${current_price:.2f}")
            except Exception as e:
                print(f"[WARNING] Could not fetch current price for {ticker}: {e}")
                # Use purchase price as fallback
                current_prices[ticker] = portfolio_purchase_info[ticker]['avg_purchase_price']
        
        print(f"[INFO] ===== Generating Recommendations =====")
        print(f"[INFO] Portfolio tickers: {tickers}")
        print(f"[INFO] Portfolio weights: {portfolio_weights}")
        
        # Load extracted filing data for portfolio tickers
        print(f"[INFO] Loading filing data for {len(tickers)} portfolio companies...")
        filings_data = load_portfolio_filings(tickers)
        
        # Load directives from S3
        print(f"[INFO] Loading regulatory directives from S3...")
        directives_data = load_directives_for_recommendations()
        if directives_data:
            print(f"[INFO] Loaded {len(directives_data)} directive(s) for analysis")
        
        # Provide detailed feedback about filing data availability
        missing_tickers = [t for t in tickers if t not in filings_data]
        if missing_tickers:
            print(f"[WARNING] No filing data found for: {missing_tickers}")
        
        # Even without filing data, show basic recommendations based on portfolio structure
        if not filings_data:
            result = "## 📊 Portfolio Recommendations\n\n"
            result += f"**⚠️ No SEC filing data found for portfolio companies.**\n\n"
            result += f"**Your Portfolio:**\n"
            for ticker in tickers:
                weight_pct = portfolio_weights.get(ticker, 0) * 100
                ticker_rows = portfolio_df[portfolio_df['Ticker'] == ticker]
                total_shares = ticker_rows['Quantity'].sum()
                result += f"- **{ticker}**: {weight_pct:.2f}% ({total_shares} shares)\n"
            
            result += f"\n**To get AI-powered recommendations:**\n"
            result += f"1. Click 'Load Portfolio' from S3 dropdown (this automatically extracts SEC filings)\n"
            result += f"2. Wait for filing extraction to complete\n"
            result += f"3. Click this button again\n\n"
            result += f"**Expected filing location:** `data/fillings/{{ticker}}/` in S3\n"
            result += f"**Portfolio tickers needing data:** {', '.join(tickers)}\n\n"
            result += f"**💡 Alternative:** You can also manually review SEC filings in the Data Explorer tab."
            return result
        
        # Extract relevant sections for each ticker (token optimization)
        filing_sections = {}
        for ticker, filing_data in filings_data.items():
            sections = get_relevant_sections_for_analysis(filing_data, max_chars_per_section=2000)
            if sections:
                filing_sections[ticker] = sections
                print(f"[INFO] Loaded sections for {ticker}: {list(sections.keys())}")
            else:
                print(f"[WARNING] No relevant sections found for {ticker}")
        
        if not filing_sections:
            result = "## 📊 Portfolio Recommendations\n\n"
            result += f"**⚠️ No relevant sections found in extracted filing data.**\n\n"
            result += f"**Your Portfolio:**\n"
            for ticker in tickers:
                weight_pct = portfolio_weights.get(ticker, 0) * 100
                result += f"- **{ticker}**: {weight_pct:.2f}%\n"
            
            result += f"\n**Files found in S3:** {', '.join(filings_data.keys())}\n"
            result += f"**Issue:** Extracted filings may not contain expected sections.\n\n"
            result += f"**Next Steps:**\n"
            result += f"1. Check if SEC filing extraction completed successfully\n"
            result += f"2. Review files in `data/fillings/{{ticker}}/` in S3\n"
            result += f"3. Try loading the portfolio again to re-extract filings"
            return result
        
        print(f"[INFO] Generating recommendations using LLM for {len(filing_sections)} companies...")
        print(f"[INFO] Portfolio weights: {portfolio_weights}")
        print(f"[INFO] Filing sections available: {list(filing_sections.keys())}")
        
        # Generate recommendations using LLM
        try:
            recommendations = llm_client.generate_portfolio_recommendations_from_filings(
                portfolio_tickers=list(filing_sections.keys()),
                portfolio_weights=portfolio_weights,
                filing_sections=filing_sections,
                directives_data=directives_data if directives_data else {}
            )
            
            print(f"[INFO] Recommendations received: {len(recommendations.get('recommendations', []))} items")
            print(f"[INFO] Overall strategy: {recommendations.get('overall_strategy', 'N/A')[:100]}")
        except Exception as e:
            print(f"[ERROR] Error in generate_portfolio_recommendations_from_filings: {e}")
            import traceback
            traceback.print_exc()
            return (
                f"❌ **Error generating recommendations:** {str(e)}\n\n"
                "**Please check:**\n"
                "1. LLM API key is configured in .env file\n"
                "2. Filing data exists for your portfolio tickers\n"
                "3. Check console logs for detailed error information"
            )
        
        # Format output with priority sorting
        result = "## 📊 Portfolio Recommendations\n\n"
        result += f"**Overall Strategy:** {recommendations.get('overall_strategy', 'N/A')}\n\n"
        result += f"**Risk Assessment:** {recommendations.get('risk_assessment', 'N/A')}\n\n"
        
        recs = recommendations.get('recommendations', [])
        if not recs:
            # Show helpful diagnostic information
            result = "## 📊 Portfolio Recommendations\n\n"
            result += f"**Overall Strategy:** {recommendations.get('overall_strategy', 'N/A')}\n\n"
            result += f"**Risk Assessment:** {recommendations.get('risk_assessment', 'N/A')}\n\n"
            result += "**⚠️ No specific recommendations generated.**\n\n"
            result += "**Diagnostic Information:**\n\n"
            result += f"- **Filing data loaded:** {len(filings_data)}/{len(tickers)} companies\n"
            if len(filings_data) < len(tickers):
                missing = [t for t in tickers if t not in filings_data]
                result += f"- **Missing filing data for:** {', '.join(missing)}\n"
                result += f"- **Action needed:** Load a portfolio from S3 to extract SEC filings\n\n"
            else:
                result += "- ✅ All portfolio companies have filing data\n\n"
            
            result += "**Possible reasons:**\n"
            result += "1. **LLM API not configured** (add OPENAI_API_KEY or PERPLEXITY_API_KEY to .env)\n"
            result += "2. API call failed (check console logs for errors)\n"
            result += "3. Filing sections may be empty (check extraction status)\n\n"
            result += "**Next Steps:**\n"
            result += "1. **Configure API Key:** Add one of these to your `.env` file:\n"
            result += "   - `OPENAI_API_KEY=sk-your-key-here` (get from https://platform.openai.com/api-keys)\n"
            result += "   - `PERPLEXITY_API_KEY=pplx-your-key-here` (get from https://www.perplexity.ai/settings/api)\n"
            result += "2. Ensure SEC filing extraction completed successfully\n"
            result += "3. Review console output for detailed error messages\n"
            result += "4. Click this button again after configuring the API key\n\n"
            result += f"**Note:** Without an API key, you'll see basic portfolio information but not AI-powered analysis."
            return result
        
        # Group by priority
        critical = [r for r in recs if r.get('priority', 'low').lower() in ['critical', 'high']]
        medium = [r for r in recs if r.get('priority', 'low').lower() == 'medium']
        low = [r for r in recs if r.get('priority', 'low').lower() == 'low']
        
        # Collect all recommended weights and normalize to 100%
        recommended_weights = {}
        for rec in recs:
            ticker = rec.get('ticker')
            if ticker:
                rec_weight = rec.get('recommended_weight', portfolio_weights.get(ticker, 0))
                if isinstance(rec_weight, str):
                    rec_weight = float(rec_weight.replace('%', ''))
                rec_weight = float(rec_weight)
                if rec_weight > 1:
                    rec_weight = rec_weight / 100
                recommended_weights[ticker] = max(0, rec_weight)
        
        total_recommended = sum(recommended_weights.values())
        sector_diversification = recommendations.get('sector_diversification', [])
        
        # Normalize weights
        if total_recommended > 0:
            if total_recommended < 0.90 and not sector_diversification:
                cash_suggestion = 1.0 - total_recommended
                stock_target = 1.0 - cash_suggestion
                if stock_target > 0:
                    normalization_factor = stock_target / total_recommended
                    normalized_weights = {t: w * normalization_factor for t, w in recommended_weights.items()}
                else:
                    normalized_weights = recommended_weights.copy()
                    cash_suggestion = 1.0 - sum(normalized_weights.values())
            else:
                normalization_factor = 1.0 / total_recommended if total_recommended > 0 else 0
                normalized_weights = {t: w * normalization_factor for t, w in recommended_weights.items()}
                cash_suggestion = 0
        else:
            normalized_weights = portfolio_weights.copy()
            total_current = sum(normalized_weights.values())
            if total_current > 0:
                normalization_factor = 1.0 / total_current
                normalized_weights = {t: w * normalization_factor for t, w in normalized_weights.items()}
            cash_suggestion = 0
        
        # Critical/High Priority
        if critical:
            result += "### 🔴 Critical / High Priority\n\n"
            for rec in critical:
                ticker = rec.get('ticker', 'N/A')
                action = rec.get('action', 'hold').upper()
                current_weight = portfolio_weights.get(ticker, 0) * 100
                original_rec_weight = rec.get('recommended_weight', current_weight)
                if isinstance(original_rec_weight, str):
                    original_rec_weight = float(original_rec_weight.replace('%', ''))
                original_rec_weight = float(original_rec_weight)
                if original_rec_weight > 1:
                    original_rec_weight = original_rec_weight / 100
                normalized_rec_weight = normalized_weights.get(ticker, current_weight / 100) * 100
                reason = rec.get('reason', 'No reason provided')
                
                result += f"**{action} {ticker}** (Priority: {rec.get('priority', 'high').upper()})\n"
                result += f"- Current: {current_weight:.2f}% → Recommended: {normalized_rec_weight:.2f}%\n"
                result += f"- Reason: {reason}\n\n"
        
        # Medium Priority
        if medium:
            result += "### 🟡 Medium Priority\n\n"
            for rec in medium:
                ticker = rec.get('ticker', 'N/A')
                action = rec.get('action', 'hold').upper()
                current_weight = portfolio_weights.get(ticker, 0) * 100
                normalized_rec_weight = normalized_weights.get(ticker, current_weight / 100) * 100
                reason = rec.get('reason', 'No reason provided')
                
                result += f"**{action} {ticker}**\n"
                result += f"- Current: {current_weight:.2f}% → Recommended: {normalized_rec_weight:.2f}%\n"
                result += f"- Reason: {reason}\n\n"
        
        # Low Priority
        if low:
            result += "### 🟢 Low Priority\n\n"
            for rec in low:
                ticker = rec.get('ticker', 'N/A')
                action = rec.get('action', 'hold').upper()
                current_weight = portfolio_weights.get(ticker, 0) * 100
                normalized_rec_weight = normalized_weights.get(ticker, current_weight / 100) * 100
                reason = rec.get('reason', 'No reason provided')
                
                result += f"**{action} {ticker}**\n"
                result += f"- Current: {current_weight:.2f}% → Recommended: {normalized_rec_weight:.2f}%\n"
                result += f"- Reason: {reason}\n\n"
        
        # Add normalized portfolio summary table
        result += f"\n---\n\n"
        result += f"### 📋 **Normalized Portfolio Allocation**\n\n"
        result += f"| Ticker | Current | Recommended | Change |\n"
        result += f"|--------|---------|------------|--------|\n"
        
        total_normalized = 0
        for ticker in sorted(set(list(portfolio_weights.keys()) + list(normalized_weights.keys()))):
            current_w = portfolio_weights.get(ticker, 0) * 100
            normalized_w = normalized_weights.get(ticker, 0) * 100
            change = normalized_w - current_w
            total_normalized += normalized_w
            result += f"| {ticker} | {current_w:.2f}% | {normalized_w:.2f}% | {change:+.2f}% |\n"
        
        # Add sector diversification suggestions
        if sector_diversification:
            result += f"\n---\n\n"
            result += f"### 🔄 **Sector Diversification Recommendations**\n\n"
            for div_rec in sector_diversification:
                reduce_sector = div_rec.get('reduce_sector', 'Unknown Sector')
                result += f"**Reduce Exposure to: {reduce_sector}**\n\n"
                
                alternatives = div_rec.get('alternative_sectors', [])
                for alt_sector in alternatives:
                    sector_name = alt_sector.get('sector_name', 'Unknown')
                    suggested_tickers = alt_sector.get('suggested_tickers', [])
                    reasons = alt_sector.get('reasons', 'No reasons provided')
                    complement = alt_sector.get('portfolio_complement', 'No complement info')
                    
                    result += f"**→ Consider {sector_name} Sector:**\n"
                    if suggested_tickers:
                        result += f"- **Suggested Tickers:** {', '.join(suggested_tickers)}\n"
                    result += f"- **Reasons:** {reasons}\n"
                    result += f"- **Portfolio Complement:** {complement}\n\n"
        
        # Add cash if suggested
        if cash_suggestion > 0.01:
            result += f"| **Cash** | 0.00% | {cash_suggestion * 100:.2f}% | +{cash_suggestion * 100:.2f}% |\n"
            total_normalized += cash_suggestion * 100
            result += f"\n⚠️ **Cash Allocation Note:** {cash_suggestion * 100:.1f}% cash is suggested only because no better investment options were identified. "
            if sector_diversification:
                result += f"Consider the sector diversification opportunities above.\n"
            else:
                result += f"Consider exploring additional sectors for diversification.\n"
        
        result += f"\n**Total:** {total_normalized:.2f}% ✅\n"
        
        # Add Performance Analysis Section
        result += f"\n---\n\n"
        result += f"### 📈 **Current Portfolio Performance**\n\n"
        result += f"| Ticker | Purchase Price | Current Price | Gain/Loss % | Gain/Loss $ |\n"
        result += f"|--------|----------------|---------------|-------------|-------------|\n"
        
        total_current_value = 0
        total_cost_basis = 0
        
        for ticker in tickers:
            purchase_info = portfolio_purchase_info.get(ticker, {})
            avg_purchase = purchase_info.get('avg_purchase_price', 0)
            quantity = purchase_info.get('total_quantity', 0)
            current_price = current_prices.get(ticker, avg_purchase)
            
            cost_basis = avg_purchase * quantity
            current_value = current_price * quantity
            gain_loss_pct = ((current_price - avg_purchase) / avg_purchase * 100) if avg_purchase > 0 else 0
            gain_loss_dollars = current_value - cost_basis
            
            total_current_value += current_value
            total_cost_basis += cost_basis
            
            gain_indicator = "🟢" if gain_loss_dollars >= 0 else "🔴"
            result += f"| {ticker} | ${avg_purchase:.2f} | ${current_price:.2f} | {gain_indicator} {gain_loss_pct:+.2f}% | {gain_indicator} ${gain_loss_dollars:+,.2f} |\n"
        
        portfolio_gain_loss_pct = ((total_current_value - total_cost_basis) / total_cost_basis * 100) if total_cost_basis > 0 else 0
        portfolio_gain_loss_dollars = total_current_value - total_cost_basis
        gain_indicator = "🟢" if portfolio_gain_loss_dollars >= 0 else "🔴"
        
        result += f"| **TOTAL** | ${total_cost_basis:,.2f} | ${total_current_value:,.2f} | {gain_indicator} {portfolio_gain_loss_pct:+.2f}% | {gain_indicator} ${portfolio_gain_loss_dollars:+,.2f} |\n"
        
        # Add "What If" Recommended Portfolio Analysis
        result += f"\n---\n\n"
        result += f"### 🔮 **What If: Recommended Portfolio Scenario**\n\n"
        result += f"*Projection if you rebalance according to recommendations at current prices*\n\n"
        
        # Show rebalancing trades
        result += f"**Rebalancing Trades Required:**\n\n"
        trades_summary = []
        total_trades_cost = 0
        
        for ticker in sorted(set(list(portfolio_weights.keys()) + list(normalized_weights.keys()))):
            current_weight = portfolio_weights.get(ticker, 0)
            recommended_weight = normalized_weights.get(ticker, 0)
            
            if abs(recommended_weight - current_weight) > 0.001:
                current_price = current_prices.get(ticker, portfolio_purchase_info.get(ticker, {}).get('avg_purchase_price', 0))
                current_shares = portfolio_purchase_info.get(ticker, {}).get('total_quantity', 0)
                current_position_value = total_current_value * current_weight
                recommended_position_value = total_current_value * recommended_weight
                
                value_change = recommended_position_value - current_position_value
                
                if recommended_weight > current_weight:
                    action = "BUY"
                    shares_to_buy = (recommended_position_value - current_position_value) / current_price if current_price > 0 else 0
                    trade_cost = shares_to_buy * current_price
                    trades_summary.append({
                        'ticker': ticker,
                        'action': action,
                        'shares': shares_to_buy,
                        'value': trade_cost,
                        'new_weight': recommended_weight * 100
                    })
                    total_trades_cost += trade_cost
                elif recommended_weight < current_weight:
                    action = "SELL"
                    shares_to_sell = (current_position_value - recommended_position_value) / current_price if current_price > 0 else 0
                    trade_proceeds = shares_to_sell * current_price
                    trades_summary.append({
                        'ticker': ticker,
                        'action': action,
                        'shares': shares_to_sell,
                        'value': trade_proceeds,
                        'new_weight': recommended_weight * 100
                    })
                    total_trades_cost -= trade_proceeds
        
        if trades_summary:
            result += f"| Action | Ticker | Shares | Value ($) | New Weight |\n"
            result += f"|--------|--------|--------|-----------|------------|\n"
            for trade in trades_summary:
                result += f"| {trade['action']} | {trade['ticker']} | {trade['shares']:.2f} | ${trade['value']:,.2f} | {trade['new_weight']:.2f}% |\n"
            
            net_cash_flow = -total_trades_cost
            if abs(net_cash_flow) > 1:
                if net_cash_flow > 0:
                    result += f"\n*Net cash from trades: ${net_cash_flow:,.2f} (proceeds exceed purchases)*\n"
                else:
                    result += f"\n*Net cash required: ${abs(net_cash_flow):,.2f} (to execute all buys)*\n"
        else:
            result += f"*No significant rebalancing trades needed - portfolio is well-aligned with recommendations.*\n\n"
        
        recommended_portfolio_value = total_current_value
        
        if cash_suggestion > 0.01:
            cash_amount = total_current_value * cash_suggestion
            result += f"\n*Cash allocation: ${cash_amount:,.2f} ({cash_suggestion * 100:.1f}%) - to be held in portfolio*\n"
            recommended_portfolio_value = total_current_value
        
        result += f"\n**Recommended Portfolio Composition Value:** ${recommended_portfolio_value:,.2f}\n"
        result += f"**Current Portfolio Value:** ${total_current_value:,.2f}\n"
        result += f"**Cost Basis:** ${total_cost_basis:,.2f}\n\n"
        
        result += f"**Note:** This shows portfolio structure if rebalanced. Future performance depends on:\n"
        result += f"- Actual price movements of recommended holdings\n"
        result += f"- Timing of trades\n"
        result += f"- Market conditions\n"
        result += f"*The recommended allocation aims to optimize risk-adjusted returns based on current regulatory and filing analysis.*\n"
        
        result += f"\n---\n\n"
        result += f"**📊 Analysis Summary:**\n"
        result += f"- Analyzed {len(filings_data)} SEC 10-K filings\n"
        result += f"- Generated {len(recs)} recommendations\n"
        result += f"- Priority breakdown: {len(critical)} critical/high, {len(medium)} medium, {len(low)} low\n\n"
        
        # Check API key status
        api_key_status = "✅ Configured" if (os.getenv('OPENAI_API_KEY') or os.getenv('PERPLEXITY_API_KEY')) else "❌ Not configured"
        result += f"**🔑 API Status:** {api_key_status}\n"
        if api_key_status == "❌ Not configured":
            result += f"\n**💡 To enable AI-powered recommendations:**\n"
            result += f"1. Get an API key from:\n"
            result += f"   - OpenAI: https://platform.openai.com/api-keys (recommended)\n"
            result += f"   - Perplexity: https://www.perplexity.ai/settings/api\n"
            result += f"2. Add to your `.env` file:\n"
            result += f"   `OPENAI_API_KEY=sk-your-key-here`\n"
            result += f"3. Restart the application and try again\n"
        else:
            result += f"\n**✅ AI recommendations are enabled and working!**\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error generating recommendations: {str(e)}"


def generate_portfolio_recommendations(impact_json: str, portfolio_json: str) -> str:
    """Generate portfolio adjustment recommendations based on user's portfolio."""
    if not portfolio_json:
        return "⚠️ Please upload or enter your portfolio first."
    
    if not impact_json or impact_json.strip() == "":
        return "⚠️ Please analyze a regulatory document first to see recommendations."
    
    try:
        impact = json.loads(impact_json) if isinstance(impact_json, str) else impact_json
        portfolio = json.loads(portfolio_json) if isinstance(portfolio_json, str) else portfolio_json
        
        # Get recommendations using actual portfolio
        recommendations = llm_client.generate_recommendations(impact, portfolio)
        
        # Match recommendations with portfolio holdings
        portfolio_tickers = {h['ticker']: h['weight'] for h in portfolio.get('holdings', [])}
        
        # Format output
        result = "## Portfolio Recommendations\n\n"
        result += f"**Overall Strategy:** {recommendations.get('overall_strategy', 'N/A')}\n"
        result += f"**Risk Assessment:** {recommendations.get('risk_assessment', 'N/A')}\n\n"
        
        if recommendations.get('recommendations'):
            result += "**Recommended Actions:**\n\n"
            for rec in recommendations['recommendations']:
                ticker = rec.get('ticker', 'N/A')
                current_weight = portfolio_tickers.get(ticker, 0) * 100
                result += f"**{rec.get('action', 'N/A').upper()}** {ticker}\n"
                result += f"  Current: {current_weight:.2f}% → Recommended: {rec.get('recommended_weight', 'N/A')}%\n"
                result += f"  Reason: {rec.get('reason', 'N/A')}\n\n"
        else:
            result += "No specific recommendations at this time. Monitor regulatory developments.\n"
        
        return result
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def run_simulation(portfolio_json: str) -> str:
    """Run portfolio simulation with regulatory impacts."""
    if not portfolio_json:
        return "⚠️ Please upload or enter your portfolio first."
    
    try:
        portfolio = json.loads(portfolio_json) if isinstance(portfolio_json, str) else portfolio_json
        
        # Define scenarios
        scenarios = [
            {"name": "Baseline (No Regulatory Impact)"},
            {"name": "Regulatory Impact Applied"},
            {"name": "Moderate Compliance Costs"}
        ]
        
        simulation_results = llm_client.run_simulation(portfolio, scenarios)
        
        # Format output
        result = "## Portfolio Simulation Results\n\n"
        result += f"**Portfolio Holdings:** {len(portfolio.get('holdings', []))} positions\n\n"
        
        for scenario in simulation_results.get('scenarios', []):
            result += f"### {scenario.get('name', 'Unknown')}\n"
            result += f"- Expected Return: {scenario.get('expected_return', 'N/A')}\n"
            result += f"- Portfolio Value Change: {scenario.get('portfolio_value_change', 'N/A')}\n"
            
            if scenario.get('risk_metrics'):
                risk = scenario['risk_metrics']
                result += f"- Volatility: {risk.get('volatility', 'N/A')}\n"
                result += f"- Sharpe Ratio: {risk.get('sharpe_ratio', 'N/A')}\n"
            
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error running simulation: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Regulatory Impact Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Regulatory Impact Analyzer")
    gr.Markdown("Analyze regulatory documents and evaluate their financial impact on S&P 500 companies.")
    
    with gr.Tabs():
        # Tab 1: Document Upload/Selection
        with gr.Tab("📄 Documents"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Select or Upload Document")
                    document_selector = gr.Dropdown(
                        choices=get_directive_list(),
                        label="Available Regulatory Documents (from S3)",
                        interactive=True
                    )
                    document_file = gr.File(label="Or Upload New Document")
                    load_btn = gr.Button("Load Document", variant="primary")
                    
                with gr.Column():
                    gr.Markdown("### Document Preview")
                    document_metadata = gr.Markdown()
                    document_preview = gr.Textbox(
                        label="Document Text",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
            
            def load_document_wrapper(path):
                if path:
                    return load_document(path)
                else:
                    return "", "Please select a document from the dropdown or upload a new file."
            
            load_btn.click(
                fn=load_document_wrapper,
                inputs=document_selector,
                outputs=[document_preview, document_metadata]
            )
        
        # Tab 2: Analysis
        with gr.Tab("🔍 Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Document Analysis")
                    analyze_btn = gr.Button("Extract Entities", variant="primary")
                    entities_output = gr.Markdown(label="Extracted Entities")
                    
                with gr.Column():
                    gr.Markdown("### Impact Evaluation")
                    evaluate_btn = gr.Button("Evaluate Impact on S&P 500", variant="primary")
                    impact_output = gr.Markdown(label="Impact Analysis")
            
            analyze_btn.click(
                fn=analyze_document,
                inputs=document_preview,
                outputs=entities_output
            )
            
            evaluate_btn.click(
                fn=evaluate_impact,
                inputs=entities_output,
                outputs=impact_output
            )
        
        # Tab 3: Portfolio
        with gr.Tab("💼 Portfolio"):
            # Portfolio state (stores DataFrame)
            portfolio_df_state = gr.State(value=pd.DataFrame())
            
            with gr.Row():
                # Left column: Portfolio Management
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Manage Your Portfolio")
                    
                    # Load existing portfolio
                    with gr.Accordion("📂 Load Existing Portfolio", open=False):
                        with gr.Row():
                            portfolio_dropdown = gr.Dropdown(
                                choices=["Select a portfolio..."] + list_portfolios_in_s3(),
                                label="Select Portfolio from S3",
                                value="Select a portfolio...",
                                interactive=True,
                                scale=4
                            )
                            refresh_dropdown_btn = gr.Button("🔄", variant="secondary", size="sm", scale=1, min_width=50)
                        load_portfolio_btn = gr.Button("Load Portfolio", variant="secondary")
                        load_portfolio_status = gr.Markdown()
                    
                    # Add new stock entry
                    gr.Markdown("### ➕ Add Stock to Portfolio")
                    gr.Markdown("**Provide either date OR price** (date will fetch historical price automatically)")
                    with gr.Row():
                        portfolio_ticker = gr.Textbox(
                            label="Stock Symbol",
                            placeholder="AAPL",
                            scale=1
                        )
                        portfolio_quantity = gr.Number(
                            label="Number of Shares",
                            value=0,
                            precision=0,
                            scale=1
                        )
                    with gr.Row():
                        portfolio_date = gr.Textbox(
                            label="Date Bought (YYYY-MM-DD) - Optional",
                            placeholder="2024-01-15",
                            scale=1
                        )
                        portfolio_price = gr.Number(
                            label="Price per Share ($) - Optional",
                            value=0.0,
                            scale=1
                        )
                    
                    add_stock_btn = gr.Button("Add Stock", variant="primary")
                    add_stock_status = gr.Markdown()
                    
                    # Save to S3
                    gr.Markdown("### 💾 Save Portfolio to S3")
                    portfolio_name_input = gr.Textbox(
                        label="Portfolio Name (optional - will auto-generate if empty or duplicate)",
                        placeholder="my_portfolio",
                        value=""
                    )
                    save_to_s3_btn = gr.Button("💾 Save Portfolio to S3", variant="primary")
                    save_to_s3_status = gr.Markdown()
                
                # Right column: Portfolio Display
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Your Portfolio Holdings")
                    portfolio_display = gr.Dataframe(
                        label="Portfolio",
                        headers=["Ticker", "Price", "Quantity", "Date_Bought"],
                        interactive=False,
                        wrap=True
                    )
                    
                    portfolio_summary = gr.Markdown()
                    
                    # Analysis section
                    with gr.Accordion("📈 Portfolio Analysis & Recommendations", open=True):
                        gr.Markdown("""
                        **AI-Powered Portfolio Recommendations**
                        
                        Analyzes SEC 10-K filings and regulatory directives for your portfolio companies to provide:
                        - Risk assessment based on filing data and regulatory compliance
                        - Actionable recommendations (Buy/
                