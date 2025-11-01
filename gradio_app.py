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
            
            # Load data when tab is opened
            demo.load(
                fn=lambda: (load_sp500_data(), load_stock_performance()),
                outputs=[sp500_display, performance_display]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

