"""Main Gradio application for Regulatory Impact Analyzer."""
import gradio as gr
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json

# Import utility modules
from utils.s3_utils import (
    get_sp500_companies,
    get_stock_performance,
    get_available_directives,
    get_available_filings,
    read_file_from_s3
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


def save_portfolio_to_s3_handler(portfolio_df: pd.DataFrame) -> str:
    """
    Save current portfolio to S3.
    
    Returns:
        Status message
    """
    if portfolio_df is None or len(portfolio_df) == 0:
        return "⚠️ Portfolio is empty. Add some stocks first."
    
    try:
        success, error = save_portfolio_to_s3(portfolio_df)
        
        if success:
            return f"✅ Portfolio saved to S3 successfully! ({len(portfolio_df)} holdings)"
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
                    # Check in data/filings/{ticker}/ for extracted JSON files
                    from utils.s3_utils import list_files_in_s3
                    extracted_prefix = f"data/filings/{ticker}/"
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
            
            status_msg += f"\n💾 Extracted data location: `data/filings/`\n"
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
                    gr.Markdown("### 📝 Manage Your Portfolio")
                    
                    # Load existing portfolio
                    with gr.Accordion("📂 Load Existing Portfolio", open=False):
                        portfolio_dropdown = gr.Dropdown(
                            choices=["Select a portfolio..."] + list_portfolios_in_s3(),
                            label="Select Portfolio from S3",
                            value="Select a portfolio...",
                            interactive=True
                        )
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
                    with gr.Accordion("📈 Analysis & Recommendations", open=False):
                        gr.Markdown("Generate recommendations based on regulatory impact analysis.")
                        recommend_btn = gr.Button("Generate Recommendations", variant="primary")
                        recommendations_output = gr.Markdown(label="Recommendations")
                        
                        simulate_btn = gr.Button("Run Simulation", variant="secondary")
                        simulation_output = gr.Markdown(label="Simulation Results")
            
            # Event handlers
            # Load portfolio from dropdown
            load_portfolio_btn.click(
                fn=load_portfolio_from_s3_handler,
                inputs=portfolio_dropdown,
                outputs=[portfolio_display, load_portfolio_status]
            ).then(
                fn=lambda df: df,  # Update state
                inputs=portfolio_display,
                outputs=portfolio_df_state
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
                fn=lambda df: f"## Portfolio Summary\n\n**Total Holdings:** {len(df)} positions\n\n**Total Shares:** {df['Quantity'].sum() if len(df) > 0 else 0}\n\n**Total Cost:** ${(df['Price'] * df['Quantity']).sum():,.2f}" if len(df) > 0 else "Portfolio is empty.",
                inputs=portfolio_df_state,
                outputs=portfolio_summary
            )
            
            # Save portfolio to S3
            save_to_s3_btn.click(
                fn=save_portfolio_to_s3_handler,
                inputs=portfolio_df_state,
                outputs=save_to_s3_status
            ).then(
                fn=lambda: ["Select a portfolio..."] + list_portfolios_in_s3(),
                inputs=None,
                outputs=portfolio_dropdown
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

