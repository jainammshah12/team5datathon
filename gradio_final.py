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
            return pd.DataFrame({
                "Error": [f"⚠️ Failed to load stock performance data: {str(e)}"],
                "Solution": ["Please check AWS credentials in .env file or ensure local data/ folder exists"]
            })
    return _stock_performance

def get_directive_list() -> List[str]:
    """Get list of available directives from S3."""
    try:
        directives = get_available_directives()
        return [d for d in directives if '.' in d.split('/')[-1] and 'README' not in d]
    except Exception as e:
        return [f"Error loading directives: {str(e)}"]

def load_directives_for_recommendations() -> Dict[str, Dict[str, str]]:
    """Load regulatory directives from S3 for use in portfolio recommendations."""
    directives_data = {}
    try:
        directive_files = get_available_directives()
        raw_files = [d for d in directive_files if d.endswith(('.html', '.xml', '.htm'))]
        json_files = [d for d in directive_files if d.endswith('.json')]
        
        for directive_path in json_files[:3]:
            try:
                content = read_file_from_s3(directive_path)
                if content:
                    directive_data = json.loads(content)
                    directive_name = directive_path.split('/')[-1].replace('.json', '').replace('extracted_', '')
                    directives_data[directive_name] = directive_data
                    print(f"[INFO] Loaded pre-extracted directive: {directive_name}")
            except Exception as e:
                print(f"[WARNING] Could not load JSON directive {directive_path}: {e}")
                continue
        
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
                for directive_path in raw_files[:3]:
                    if len(directives_data) >= 5:
                        break
                    
                    try:
                        print(f"[INFO] Processing raw directive: {directive_path}")
                        content = read_file_from_s3(directive_path)
                        if not content:
                            continue
                        
                        is_xml = is_xml_content(content)
                        text = extract_full_text_from_html(content, is_xml=is_xml)
                        
                        try:
                            language, _ = detect_language(text)
                        except:
                            language = 'en'
                        
                        sections = extract_sections_from_directive(content, is_xml=is_xml, language=language)
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


# FIXED: Proper document loading with summarization
def load_document_wrapper(dropdown_selection: str, uploaded_file) -> Tuple[str, str]:
    """
    Load document from either S3 dropdown or uploaded file.
    Returns (raw_text, summary_markdown).
    """
    document_path = None
    
    # Check if uploaded file provided
    if uploaded_file is not None:
        try:
            file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate summary
            summary, metadata = llm_client.summarize_document(content)
            
            # Format metadata display
            meta_display = "## Document Summary\n\n"
            meta_display += f"{summary}\n\n"
            meta_display += "### Metadata\n\n"
            meta_display += f"**Type:** {metadata.get('document_type', 'Unknown')}\n"
            meta_display += f"**Jurisdiction:** {metadata.get('jurisdiction', 'Unknown')}\n"
            
            if metadata.get('effective_date'):
                meta_display += f"**Effective Date:** {metadata['effective_date']}\n"
            if metadata.get('compliance_deadline'):
                meta_display += f"**Compliance Deadline:** {metadata['compliance_deadline']}\n"
            
            if metadata.get('key_topics'):
                meta_display += f"\n**Key Topics:** {', '.join(metadata['key_topics'])}\n"
            
            if metadata.get('affected_sectors'):
                meta_display += f"**Affected Sectors:** {', '.join(metadata['affected_sectors'])}\n"
            
            return content, meta_display
            
        except Exception as e:
            return "", f"Error loading uploaded file: {str(e)}"
    
    # Check dropdown selection
    if dropdown_selection and dropdown_selection != "Select a document...":
        document_path = dropdown_selection
    
    if not document_path:
        return "", "Please select a document from the dropdown or upload a new file."
    
    try:
        content = read_file_from_s3(document_path)
        base_metadata = extract_metadata(document_path)
        
        # Extract text based on file type
        if document_path.endswith('.html'):
            text = extract_text_from_html(content)
        elif document_path.endswith('.xml'):
            text = extract_text_from_xml(content)
        else:
            text = content
        
        text = clean_text(text)
        
        # Generate AI summary
        summary, ai_metadata = llm_client.summarize_document(text)
        
        # Format display
        meta_display = "## Document Summary\n\n"
        meta_display += f"{summary}\n\n"
        meta_display += "### File Information\n\n"
        meta_display += f"**File:** {base_metadata['filename']}\n"
        
        if base_metadata['date']:
            meta_display += f"**Date:** {base_metadata['date']}\n"
        if base_metadata['type']:
            meta_display += f"**Type:** {base_metadata['type']}\n"
        if base_metadata['ticker']:
            meta_display += f"**Ticker:** {base_metadata['ticker']}\n"
        
        meta_display += "\n### AI-Extracted Metadata\n\n"
        meta_display += f"**Document Type:** {ai_metadata.get('document_type', 'Unknown')}\n"
        meta_display += f"**Jurisdiction:** {ai_metadata.get('jurisdiction', 'Unknown')}\n"
        
        if ai_metadata.get('effective_date'):
            meta_display += f"**Effective Date:** {ai_metadata['effective_date']}\n"
        if ai_metadata.get('compliance_deadline'):
            meta_display += f"**Compliance Deadline:** {ai_metadata['compliance_deadline']}\n"
        
        if ai_metadata.get('key_topics'):
            meta_display += f"\n**Key Topics:**\n"
            for topic in ai_metadata['key_topics']:
                meta_display += f"- {topic}\n"
        
        if ai_metadata.get('affected_sectors'):
            meta_display += f"\n**Affected Sectors:**\n"
            for sector in ai_metadata['affected_sectors']:
                meta_display += f"- {sector}\n"
        
        return text, meta_display
        
    except Exception as e:
        return "", f"Error loading document: {str(e)}"


def analyze_document(document_text: str) -> str:
    """Analyze document and extract entities."""
    if not document_text:
        return "Please load a document first."
    
    try:
        entities = llm_client.extract_entities(document_text)
        
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


# FIXED: Evaluate impact accepts raw document text
def evaluate_impact(document_text: str) -> str:
    """Evaluate financial impact on S&P 500 companies."""
    if not document_text:
        return "Please load a document first."
    
    try:
        # First extract entities
        entities = llm_client.extract_entities(document_text)
        
        # Load S&P 500 data
        sp500_data = load_sp500_data()
        
        if "Error" in sp500_data.columns:
            return "⚠️ Cannot load S&P 500 data. Please check configuration."
        
        # Convert DataFrame to dict for LLM
        companies_dict = sp500_data.to_dict('records')[:50]
        
        # Analyze impact
        impact = llm_client.analyze_impact(entities, companies_dict)
        
        # Format output
        result = "## Impact Analysis\n\n"
        result += f"**Most Affected Sector:** {impact['sector_summary'].get('most_affected_sector', 'Unknown')}\n"
        result += f"**Overall Impact:** {impact['sector_summary'].get('overall_impact', 'Pending analysis')}\n\n"
        
        if impact.get('affected_companies'):
            result += "**Affected Companies:**\n\n"
            for company in impact['affected_companies'][:10]:
                result += f"**{company.get('ticker', 'N/A')}** - {company.get('company_name', 'N/A')}\n"
                result += f"  Impact: {company.get('impact_level', 'Unknown')} ({company.get('impact_type', 'Unknown')})\n"
                result += f"  Rationale: {company.get('rationale', 'N/A')}\n\n"
        
        return result
    except Exception as e:
        return f"Error evaluating impact: {str(e)}"


def fetch_price_for_date(ticker: str, date: str) -> Tuple[Optional[float], Optional[str]]:
    """Fetch historical stock price for a given date."""
    try:
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        start_date = (date_obj - timedelta(days=5)).strftime('%Y-%m-%d')
        end_date = (date_obj + timedelta(days=5)).strftime('%Y-%m-%d')
        
        data = fetch_daily_stock_data(ticker, start_date, end_date)
        
        if data.empty:
            return None, f"No data found for {ticker} around {date}"
        
        if data['Date'].dtype == 'object':
            data['Date'] = pd.to_datetime(data['Date'])
        target_date = pd.to_datetime(date)
        
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
    """Add a stock entry to the portfolio."""
    if not ticker or not quantity:
        return "⚠️ Please provide ticker and quantity", current_portfolio_df
    
    has_date = date_bought and date_bought.strip()
    has_price = price and price > 0
    
    if not has_date and not has_price:
        return "⚠️ Please provide either purchase date OR price", current_portfolio_df
    
    try:
        ticker = ticker.upper().strip()
        quantity = int(quantity)
        
        if quantity <= 0:
            return "❌ Quantity must be greater than 0", current_portfolio_df
        
        if has_date:
            from datetime import datetime
            try:
                date_obj = datetime.strptime(date_bought.strip(), '%Y-%m-%d')
                if date_obj > datetime.now():
                    return "❌ Date cannot be in the future", current_portfolio_df
                if date_obj.year < 1990:
                    return "❌ Date seems too old. Please check.", current_portfolio_df
            except ValueError:
                return "❌ Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-15)", current_portfolio_df
        
        final_price = None
        if has_date:
            fetched_price, error = fetch_price_for_date(ticker, date_bought)
            if error:
                return f"❌ {error}", current_portfolio_df
            final_price = fetched_price
            final_date = date_bought
        else:
            final_price = float(price)
            from datetime import datetime
            final_date = datetime.now().strftime('%Y-%m-%d')
        
        if final_price <= 0:
            return "❌ Price must be greater than 0", current_portfolio_df
        
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
    """Save current portfolio to S3 with optional name."""
    if portfolio_df is None or len(portfolio_df) == 0:
        return "⚠️ Portfolio is empty. Add some stocks first."
    
    try:
        if not portfolio_name or not portfolio_name.strip():
            portfolio_name = "portfolio"
        
        portfolio_name = "".join(c for c in portfolio_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not portfolio_name:
            portfolio_name = "portfolio"
        
        existing_portfolios = list_portfolios_in_s3()
        base_filename = f"{portfolio_name}.csv"
        filename = base_filename
        
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
    """Load portfolio from S3 and extract SEC filing data for each ticker."""
    if not filename or filename == "Select a portfolio...":
        return pd.DataFrame(), ""
    
    try:
        df, error = load_portfolio_from_s3(filename)
        
        if df is not None and len(df) > 0:
            status_msg = f"✅ Loaded portfolio '{filename}' with {len(df)} holdings\n\n"
            status_msg += "📄 **SEC Filing Extraction:**\n"
            
            portfolio_name = filename.replace('.csv', '').replace('portfolio_', '')
            
            print(f"[INFO] Starting SEC filing extraction for portfolio companies...")
            unique_tickers = df['Ticker'].unique()
            
            extraction_count = 0
            skipped_count = 0
            
            for ticker in unique_tickers:
                try:
                    extracted_prefix = f"data/fillings/{ticker}/"
                    existing_extractions = list_files_in_s3(extracted_prefix)
                    
                    existing_10k_files = [f for f in existing_extractions if ('10_k' in f.lower() or '10k' in f.lower()) and f.endswith('.json')]
                    
                    if existing_10k_files:
                        print(f"[INFO] Extracted data already exists for {ticker}, skipping extraction")
                        status_msg += f"✓ {ticker}: Using existing extraction\n"
                        skipped_count += 1
                        continue
                    
                    filings = get_available_filings(ticker)
                    
                    if filings:
                        filing_10k = [f for f in filings if '10k' in f.lower()]
                        
                        if filing_10k:
                            filing_path = filing_10k[0]
                            print(f"[INFO] Found 10-K filing for {ticker}: {filing_path}")
                            
                            try:
                                html_content = read_file_from_s3(filing_path)
                                
                                if html_content:
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
    """Handle CSV file upload for portfolio."""
    if file is None:
        return "❌ No file uploaded", pd.DataFrame(), ""
    
    try:
        file_path = file if isinstance(file, str) else file.name
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        portfolio, error = parse_portfolio_csv(file_content)
        
        if error:
            return f"❌ {error}", pd.DataFrame(), ""
        
        is_valid, validation_error = validate_portfolio(portfolio)
        if not is_valid:
            return f"❌ Validation error: {validation_error}", pd.DataFrame(), ""
        
        df = portfolio_to_dataframe(portfolio)
        portfolio_json = json.dumps(portfolio)
        
        return f"✅ Portfolio uploaded successfully! {len(portfolio['holdings'])} holdings loaded.", df, portfolio_json
        
    except Exception as e:
        return f"❌ Error uploading portfolio: {str(e)}", pd.DataFrame(), ""


def upload_portfolio_manual(portfolio_text: str) -> Tuple[str, pd.DataFrame, str]:
    """Handle manual text input for portfolio."""
    if not portfolio_text or not portfolio_text.strip():
        return "⚠️ Please enter portfolio holdings", pd.DataFrame(), ""
    
    try:
        portfolio, error = parse_portfolio_manual(portfolio_text)
        
        if error:
            return f"❌ {error}", pd.DataFrame(), ""
        
        is_valid, validation_error = validate_portfolio(portfolio)
        if not is_valid:
            return f"❌ Validation error: {validation_error}", pd.DataFrame(), ""
        
        df = portfolio_to_dataframe(portfolio)
        portfolio_json = json.dumps(portfolio)
        
        return f"✅ Portfolio loaded successfully! {len(portfolio['holdings'])} holdings.", df, portfolio_json
        
    except Exception as e:
        return f"❌ Error loading portfolio: {str(e)}", pd.DataFrame(), ""


# Keep your original generate_portfolio_recommendations_from_filings function
def generate_portfolio_recommendations_from_filings(portfolio_df: pd.DataFrame) -> str:
    """
    Generate portfolio recommendations based on SEC filing data.
    Uses extracted filing sections to provide actionable recommendations.
    
    Args:
        portfolio_df: DataFrame with portfolio holdings (Ticker, Price, Quantity, Date_Bought)
    
    Returns:
        Formatted markdown string with recommendations
    """
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
        portfolio_purchase_info = {}
        for ticker in tickers:
            ticker_rows = portfolio_df[portfolio_df['Ticker'] == ticker]
            ticker_cost = (ticker_rows['Price'] * ticker_rows['Quantity']).sum()
            portfolio_weights[ticker] = ticker_cost / total_cost if total_cost > 0 else 0
            
            portfolio_purchase_info[ticker] = {
                'avg_purchase_price': ticker_cost / ticker_rows['Quantity'].sum() if ticker_rows['Quantity'].sum() > 0 else 0,
                'total_quantity': ticker_rows['Quantity'].sum(),
                'total_cost': ticker_cost,
                'earliest_date': ticker_rows['Date_Bought'].min() if 'Date_Bought' in ticker_rows.columns else None
            }
        
        # Fetch current prices
        print(f"[INFO] Fetching current prices for {len(tickers)} holdings...")
        current_prices = {}
        for ticker in tickers:
            try:
                from utils.yfinance_fetcher import fetch_stock_info
                info = fetch_stock_info(ticker)
                current_price = info.get('Current Price')
                if current_price:
                    current_prices[ticker] = current_price
            except Exception as e:
                print(f"[WARNING] Could not fetch current price for {ticker}: {e}")
                current_prices[ticker] = portfolio_purchase_info[ticker]['avg_purchase_price']
        
        print(f"[INFO] Loading filing data for {len(tickers)} portfolio companies...")
        filings_data = load_portfolio_filings(tickers)
        
        print(f"[INFO] Loading regulatory directives from S3...")
        directives_data = load_directives_for_recommendations()
        
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
            result += f"1. Click 'Load Portfolio' from S3 dropdown\n"
            result += f"2. Wait for filing extraction to complete\n"
            result += f"3. Click this button again\n"
            return result
        
        # Extract relevant sections
        filing_sections = {}
        for ticker, filing_data in filings_data.items():
            sections = get_relevant_sections_for_analysis(filing_data, max_chars_per_section=2000)
            if sections:
                filing_sections[ticker] = sections
        
        if not filing_sections:
            return "## 📊 Portfolio Recommendations\n\n**⚠️ No relevant sections found in filing data.**"
        
        # Generate recommendations
        recommendations = llm_client.generate_portfolio_recommendations_from_filings(
            portfolio_tickers=list(filing_sections.keys()),
            portfolio_weights=portfolio_weights,
            filing_sections=filing_sections,
            directives_data=directives_data if directives_data else {}
        )
        
        # Format output
        result = "## 📊 Portfolio Recommendations\n\n"
        result += f"**Overall Strategy:** {recommendations.get('overall_strategy', 'N/A')}\n\n"
        result += f"**Risk Assessment:** {recommendations.get('risk_assessment', 'N/A')}\n\n"
        
        recs = recommendations.get('recommendations', [])
        
        if not recs:
            api_key_status = "✅ Configured" if (os.getenv('OPENAI_API_KEY') or os.getenv('PERPLEXITY_API_KEY')) else "❌ Not configured"
            result += "**⚠️ No specific recommendations generated.**\n\n"
            result += f"**🔑 API Status:** {api_key_status}\n"
            if api_key_status == "❌ Not configured":
                result += "\n**💡 Configure API Key:**\n"
                result += "- Add `OPENAI_API_KEY=sk-...` to .env\n"
                result += "- Or `PERPLEXITY_API_KEY=pplx-...`\n"
            return result
        
        # Group by priority
        critical = [r for r in recs if r.get('priority', 'low').lower() in ['critical', 'high']]
        medium = [r for r in recs if r.get('priority', 'low').lower() == 'medium']
        low = [r for r in recs if r.get('priority', 'low').lower() == 'low']
        
        # Show recommendations by priority
        if critical:
            result += "### 🔴 Critical / High Priority\n\n"
            for rec in critical:
                ticker = rec.get('ticker', 'N/A')
                action = rec.get('action', 'hold').upper()
                result += f"**{action} {ticker}**\n"
                result += f"- Reason: {rec.get('reason', 'N/A')}\n\n"
        
        if medium:
            result += "### 🟡 Medium Priority\n\n"
            for rec in medium:
                ticker = rec.get('ticker', 'N/A')
                action = rec.get('action', 'hold').upper()
                result += f"**{action} {ticker}**\n"
                result += f"- Reason: {rec.get('reason', 'N/A')}\n\n"
        
        if low:
            result += "### 🟢 Low Priority\n\n"
            for rec in low:
                ticker = rec.get('ticker', 'N/A')
                action = rec.get('action', 'hold').upper()
                result += f"**{action} {ticker}**\n"
                result += f"- Reason: {rec.get('reason', 'N/A')}\n\n"
        
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
        
        recommendations = llm_client.generate_recommendations(impact, portfolio)
        
        portfolio_tickers = {h['ticker']: h['weight'] for h in portfolio.get('holdings', [])}
        
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
        
        scenarios = [
            {"name": "Baseline (No Regulatory Impact)"},
            {"name": "Regulatory Impact Applied"},
            {"name": "Moderate Compliance Costs"}
        ]
        
        simulation_results = llm_client.run_simulation(portfolio, scenarios)
        
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
        # Tab 1: Document Upload/Selection (FIXED)
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
                    gr.Markdown("### Document Summary & Metadata")
                    document_metadata = gr.Markdown(label="AI-Generated Summary")
                    gr.Markdown("### Raw Document Text (for analysis)")
                    document_preview = gr.Textbox(
                        label="Document Text",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
            
            # FIXED: Proper binding for document loading
            load_btn.click(
                fn=load_document_wrapper,
                inputs=[document_selector, document_file],
                outputs=[document_preview, document_metadata]
            )
        
        # Tab 2: Analysis (FIXED)
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
            
            # FIXED: Send raw document text instead of markdown
            evaluate_btn.click(
                fn=evaluate_impact,
                inputs=document_preview,
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
                        - Actionable recommendations (Buy/Sell/Hold) considering regulatory impacts
                        - Priority ranking (Critical/Medium/Low)
                        - Weight adjustment suggestions normalized to 100%
                        
                        **Note:** Recommendations consider both company SEC filings and regulatory directives from data/directives/ in S3.
                        """)
                        
                        filing_recommend_btn = gr.Button(
                            "🤖 Generate AI Recommendations from SEC Filings", 
                            variant="primary"
                        )
                        filing_recommendations_output = gr.Markdown(label="Recommendations")
                        
                        gr.Markdown("---")
                        gr.Markdown("**Regulatory Impact Analysis** (requires document analysis)")
                        recommend_btn = gr.Button("Generate Recommendations", variant="secondary", size="sm")
                        recommendations_output = gr.Markdown(label="Regulatory Recommendations", visible=False)
                        
                        simulate_btn = gr.Button("Run Simulation", variant="secondary", size="sm")
                        simulation_output = gr.Markdown(label="Simulation Results", visible=False)
            
            # Event handlers
            # Load portfolio from dropdown
            load_portfolio_btn.click(
                fn=load_portfolio_from_s3_handler,
                inputs=portfolio_dropdown,
                outputs=[portfolio_display, load_portfolio_status]
            ).then(
                fn=lambda df: df,
                inputs=portfolio_display,
                outputs=portfolio_df_state
            )
            
            # Refresh dropdown button
            def refresh_portfolio_dropdown():
                """Refresh the portfolio dropdown list from S3."""
                updated_choices = ["Select a portfolio..."] + list_portfolios_in_s3()
                return gr.update(choices=updated_choices, value="Select a portfolio...")
            
            refresh_dropdown_btn.click(
                fn=refresh_portfolio_dropdown,
                inputs=None,
                outputs=portfolio_dropdown
            )
            
            # Also load when dropdown changes
            portfolio_dropdown.change(
                fn=lambda filename: load_portfolio_from_s3_handler(filename) if filename and filename != "Select a portfolio..." else (pd.DataFrame(), ""),
                inputs=portfolio_dropdown,
                outputs=[portfolio_display, load_portfolio_status]
            ).then(
                fn=lambda df: df if isinstance(df, pd.DataFrame) else pd.DataFrame(),
                inputs=portfolio_display,
                outputs=portfolio_df_state
            )
            
            # Add stock to portfolio
            add_stock_btn.click(
                fn=add_stock_to_portfolio,
                inputs=[portfolio_ticker, portfolio_quantity, portfolio_date, portfolio_price, portfolio_df_state],
                outputs=[add_stock_status, portfolio_display]
            ).then(
                fn=lambda df: df,
                inputs=portfolio_display,
                outputs=portfolio_df_state
            ).then(
                fn=lambda df: f"## Portfolio Summary\n\n**Total Holdings:** {len(df)} positions\n\n**Total Shares:** {df['Quantity'].sum() if len(df) > 0 else 0}\n\n**Total Cost:** ${(df['Price'] * df['Quantity']).sum():,.2f}\n\n💡 **Note:** AI recommendations consider both SEC 10-K filings and regulatory directives from `data/directives/` in S3." if len(df) > 0 else "Portfolio is empty.",
                inputs=portfolio_df_state,
                outputs=portfolio_summary
            )
            
            # Save portfolio to S3
            def save_and_refresh(portfolio_df, portfolio_name):
                """Save portfolio and return updated dropdown choices."""
                result = save_portfolio_to_s3_handler(portfolio_df, portfolio_name)
                updated_choices = ["Select a portfolio..."] + list_portfolios_in_s3()
                return result, gr.update(choices=updated_choices, value="Select a portfolio...")
            
            save_to_s3_btn.click(
                fn=save_and_refresh,
                inputs=[portfolio_df_state, portfolio_name_input],
                outputs=[save_to_s3_status, portfolio_dropdown]
            )
            
            # AI Recommendations from SEC Filings (main feature)
            filing_recommend_btn.click(
                fn=generate_portfolio_recommendations_from_filings,
                inputs=portfolio_df_state,
                outputs=filing_recommendations_output
            )
            
            # Analysis handlers (convert portfolio_df to JSON format for compatibility)
            recommend_btn.click(
                fn=lambda df, impact: generate_portfolio_recommendations(
                    impact,
                    json.dumps({"holdings": df.to_dict('records')}) if len(df) > 0 else ""
                ) if len(df) > 0 else "⚠️ Please add stocks to your portfolio first.",
                inputs=[portfolio_df_state, impact_output],
                outputs=recommendations_output
            )
            
            simulate_btn.click(
                fn=lambda df: run_simulation(
                    json.dumps({"holdings": df.to_dict('records')}) if len(df) > 0 else ""
                ),
                inputs=portfolio_df_state,
                outputs=simulation_output
            )
        
        # Tab 4: Data Explorer
        with gr.Tab("📊 Data Explorer"):
            gr.Markdown("### S&P 500 Companies")
            sp500_display = gr.Dataframe(
                label="S&P 500 Composition",
                interactive=False
            )
            
            gr.Markdown("### Stock Performance")
            performance_display = gr.Dataframe(
                label="Stock Performance Data",
                interactive=False
            )
            
            # Load data when tab is opened
            demo.load(
                fn=lambda: (load_sp500_data(), load_stock_performance()),
                outputs=[sp500_display, performance_display]
            )

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Regulatory Impact Analyzer")
    print("="*60)
    print(f"📍 Access the app at: http://localhost:7860")
    print(f"💡 Note: Use 'localhost' not '0.0.0.0' in your browser")
    print("="*60)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)