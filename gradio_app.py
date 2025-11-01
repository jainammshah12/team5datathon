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
    check_file_exists_in_s3,
    upload_file_to_s3
)
from utils.document_processor import (
    extract_text_from_html,
    extract_text_from_xml,
    clean_text,
    extract_metadata
)
from llm.llm_client import get_llm_client

# Initialize LLM client
llm_client = get_llm_client()

# Global cache for data
_sp500_data = None
_stock_performance = None

def load_sp500_data() -> pd.DataFrame:
    """Load S&P 500 companies data from S3 (cached)."""
    global _sp500_data
    if _sp500_data is None:
        try:
            _sp500_data = get_sp500_companies()
        except Exception as e:
            return pd.DataFrame({
                "Error": [f"⚠️ Failed to load S&P 500 data from S3: {str(e)}"],
                "Solution": ["Please check AWS credentials in .env file"]
            })
    return _sp500_data

def load_stock_performance() -> pd.DataFrame:
    """Load stock performance data from S3 (cached)."""
    global _stock_performance
    if _stock_performance is None:
        try:
            _stock_performance = get_stock_performance()
        except Exception as e:
            return pd.DataFrame({
                "Error": [f"⚠️ Failed to load stock performance data from S3: {str(e)}"],
                "Solution": ["Please check AWS credentials in .env file"]
            })
    return _stock_performance

def get_directive_list() -> List[str]:
    """Get list of available directives from S3."""
    try:
        directives = get_available_directives()
        filtered = [d for d in directives if '.' in d.split('/')[-1] and 'README' not in d]
        
        if not filtered:
            return ["⚠️ No directives found in S3 bucket."]
        
        print(f"[INFO] Successfully loaded {len(filtered)} directives from S3")
        return filtered
    except ConnectionError as e:
        print(f"[ERROR] S3 connection failed: {e}")
        return [f"⚠️ S3 Error: {str(e)}"]
    except Exception as e:
        print(f"[ERROR] Could not load directives: {e}")
        return [f"⚠️ Error: {str(e)}"]

def upload_document_to_s3(file) -> Tuple[gr.Dropdown, str, str]:
    """
    Upload a document to S3 and return updated dropdown, document text, and metadata.
    
    Args:
        file: Gradio File object
        
    Returns:
        Tuple of (updated dropdown, document text, metadata)
    """
    try:
        if file is None:
            return gr.update(), "", "⚠️ No file selected for upload."
        
        # Get file details
        file_path = file.name
        file_name = os.path.basename(file_path)
        
        # Validate file type
        valid_extensions = ['.html', '.xml', '.txt', '.md']
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in valid_extensions:
            return gr.update(), "", f"⚠️ Invalid file type. Please upload: {', '.join(valid_extensions)}"
        
        print(f"[INFO] Starting upload: {file_name}")
        
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        file_size = len(file_content)
        print(f"[INFO] File size: {file_size} bytes")
        
        # Determine S3 key (upload to data/directives/)
        s3_key = f"data/directives/{file_name}"
        
        # Check if file already exists
        file_exists = check_file_exists_in_s3(s3_key)
        
        if file_exists:
            print(f"[INFO] File already exists, will be overwritten: {s3_key}")
        
        print(f"[INFO] Uploading to S3: {s3_key}")
        
        # Upload to S3 (will automatically overwrite if exists)
        upload_file_to_s3(file_content, s3_key, overwrite=True)
        
        action = "replaced" if file_exists else "uploaded"
        print(f"[INFO] ✅ File successfully {action}: {file_name}")
        
        # Refresh directive list
        print(f"[INFO] Refreshing directive list...")
        updated_directives = get_directive_list()
        
        # Load the uploaded document
        print(f"[INFO] Loading uploaded document...")
        doc_text, doc_metadata = load_document(s3_key)
        
        # Enhanced success message with metadata
        status_icon = "🔄" if file_exists else "✅"
        status_text = "replaced" if file_exists else "uploaded"
        success_msg = f"{status_icon} **Successfully {status_text}:** {file_name}\n📦 **Size:** {file_size:,} bytes\n🔗 **S3 Path:** {s3_key}\n\n{doc_metadata}"
        
        # Return updated dropdown with the new file selected
        return (
            gr.update(choices=updated_directives, value=s3_key),
            doc_text,
            success_msg
        )
        
    except Exception as e:
        error_msg = f"❌ **Upload failed:** {str(e)}\n\nPlease check:\n- AWS credentials are valid\n- You have s3:PutObject permission\n- File is not corrupted"
        print(f"[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return gr.update(), "", error_msg

def load_document(document_path: str) -> Tuple[str, str]:
    """Load and process a document from S3."""
    try:
        if not document_path:
            return "", "Please select a document from the dropdown or upload a file."
            
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
        
        # Format metadata only
        metadata_str = f"**File:** {metadata['filename']}\n"
        if metadata['date']:
            metadata_str += f"**Date:** {metadata['date']}\n"
        if metadata['type']:
            metadata_str += f"**Type:** {metadata['type']}\n"
        if metadata['ticker']:
            metadata_str += f"**Ticker:** {metadata['ticker']}\n"
        
        return text, metadata_str
    except Exception as e:
        return "", f"❌ Error loading document: {str(e)}"

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

def generate_portfolio_recommendations(impact_json: str) -> str:
    """Generate portfolio adjustment recommendations."""
    try:
        impact = json.loads(impact_json) if isinstance(impact_json, str) else impact_json
        
        # Mock portfolio (replace with actual portfolio data)
        portfolio = {
            "holdings": [
                {"ticker": "AAPL", "weight": 0.1},
                {"ticker": "MSFT", "weight": 0.08},
                {"ticker": "GOOGL", "weight": 0.07}
            ]
        }
        
        recommendations = llm_client.generate_recommendations(impact, portfolio)
        
        # Format output
        result = "## Portfolio Recommendations\n\n"
        result += f"**Overall Strategy:** {recommendations.get('overall_strategy', 'N/A')}\n"
        result += f"**Risk Assessment:** {recommendations.get('risk_assessment', 'N/A')}\n\n"
        
        if recommendations.get('recommendations'):
            result += "**Recommended Actions:**\n\n"
            for rec in recommendations['recommendations']:
                result += f"**{rec.get('action', 'N/A').upper()}** {rec.get('ticker', 'N/A')}\n"
                result += f"  Current: {rec.get('current_weight', 'N/A')}% → Recommended: {rec.get('recommended_weight', 'N/A')}%\n"
                result += f"  Reason: {rec.get('reason', 'N/A')}\n\n"
        
        return result
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def run_simulation(scenario_config: str) -> str:
    """Run portfolio simulation."""
    try:
        # Mock scenario (replace with actual input)
        scenarios = [{"name": "Baseline"}, {"name": "Regulatory Impact"}]
        portfolio = {"holdings": []}
        
        simulation_results = llm_client.run_simulation(portfolio, scenarios)
        
        # Format output
        result = "## Simulation Results\n\n"
        for scenario in simulation_results.get('scenarios', []):
            result += f"**{scenario.get('name', 'Unknown')}:**\n"
            result += f"  Expected Return: {scenario.get('expected_return', 'N/A')}\n"
            result += f"  Portfolio Value Change: {scenario.get('portfolio_value_change', 'N/A')}\n\n"
        
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
                        choices=[],  # Will be auto-populated on page load
                        label="Available Regulatory Documents (from S3)",
                        interactive=True
                    )
                    refresh_btn = gr.Button("🔄 Refresh Document List", size="sm")
                    
                    gr.Markdown("### Upload New Document")
                    document_file = gr.File(
                        label="Upload Document (.html, .xml, .txt)",
                        file_types=[".html", ".xml", ".txt", ".md"]
                    )
                    upload_btn = gr.Button("📤 Upload to S3", variant="secondary")
                    
                    load_btn = gr.Button("Load Selected Document", variant="primary")
                    
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
                """Load document from selected path."""
                if path:
                    return load_document(path)
                else:
                    return "", "⚠️ Please select a document from the dropdown."
            
            def refresh_directives():
                """Refresh the list of available directives."""
                directives = get_directive_list()
                print(f"[INFO] Refreshing dropdown with {len(directives)} directives")
                return gr.update(choices=directives, value=None)
            
            # Auto-load directives when app starts
            demo.load(
                fn=refresh_directives,
                outputs=document_selector
            )
            
            # Refresh button - manually trigger list refresh
            refresh_btn.click(
                fn=refresh_directives,
                outputs=document_selector
            )
            
            # Upload button - upload file to S3 and auto-load it
            upload_btn.click(
                fn=upload_document_to_s3,
                inputs=document_file,
                outputs=[document_selector, document_preview, document_metadata]
            )
            
            # Load document button - load selected document
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
        
        # Tab 3: Recommendations
        with gr.Tab("💼 Portfolio"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Portfolio Recommendations")
                    recommend_btn = gr.Button("Generate Recommendations", variant="primary")
                    recommendations_output = gr.Markdown(label="Recommendations")
                    
                with gr.Column():
                    gr.Markdown("### Simulation")
                    simulate_btn = gr.Button("Run Simulation", variant="primary")
                    simulation_output = gr.Markdown(label="Simulation Results")
            
            recommend_btn.click(
                fn=generate_portfolio_recommendations,
                inputs=impact_output,
                outputs=recommendations_output
            )
            
            simulate_btn.click(
                fn=run_simulation,
                inputs=gr.Textbox(value="", visible=False),
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
            
            load_data_btn = gr.Button("Load Data", variant="primary")
            
            # Load data when button is clicked
            load_data_btn.click(
                fn=lambda: (load_sp500_data(), load_stock_performance()),
                outputs=[sp500_display, performance_display]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

