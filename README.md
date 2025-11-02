# Regulatory Impact Analyzer

AI-powered tool that analyzes regulatory documents and evaluates their financial impact on S&P 500 companies.

## Team Members

- Jainam Shah
- Hubert Lefebvre
- Jose Del Portillo Neira
- Bhavya Ruparelia

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS S3

Create `.env` file with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token  # Required for temporary credentials (ASIA*)
AWS_DEFAULT_REGION=us-west-2
S3_BUCKET_NAME=your-bucket-name
```

**Note:** If your Access Key starts with `ASIA`, you need the session token. Get it with:

```bash
aws configure export-credentials --profile default
```

### 3. Launch Application

```bash
python gradio_app.py
```

Access at: **http://localhost:7860**

## 📋 Features

### 📄 Document Management

- **Upload** regulatory documents (.html, .xml, .txt, .md) to S3
- **Automatic overwrite** - Duplicate files are replaced automatically
- **Auto-load** - Uploaded documents load immediately
- **Full text view** - Complete document text (no truncation)
- All documents stored in S3 bucket at `data/directives/`

### 📑 SEC Filing Extraction (NEW)

- **Automated extraction** of key sections from 10-K/10-Q filings
- **18 critical sections** extracted:
  - Business, Risk Factors, Cybersecurity, Properties, Legal Proceedings
  - Market for Equity (buybacks/dividends), MD&A, Market Risk, Financial Statements
  - Controls & Procedures, Executive Compensation
  - Material Events (8-K): Acquisitions, Earnings, Impairments, Leadership Changes
- **Portfolio-based organization** - Files stored by portfolio in S3
- **S3 storage**: `extracted_filings/{portfolio_name}/{ticker}/`
- **Smart cleanup** - Automatically removes old extractions
- **No information loss** - All critical sections preserved
- Supports both HTML and XML formats

### 🔍 Analysis (Placeholder)

- Extract entities from regulatory documents
- Evaluate financial impact on S&P 500 companies
- Identify affected sectors and companies
- _Note: LLM integration pending - implement in `llm/llm_client.py`_

### 💼 Portfolio Management

- **Create and manage** multiple portfolios
- **Save to S3** - Portfolios stored as CSV files
- **Auto-extract SEC filings** when loading portfolio
- **Automatic filing extraction** for all portfolio companies
- Portfolio adjustment recommendations (with LLM)
- Financial impact simulations
- Risk assessment

### 📊 Data Explorer

- S&P 500 company composition
- Stock performance data
- SEC filings per company
- All data from S3 bucket

## 🏗️ Architecture

```
gradio_app.py           # Main Gradio interface
├── utils/
│   ├── s3_utils.py              # S3 operations (read/write/delete)
│   ├── sec_filing_extractor.py  # SEC filing parser (18 sections)
│   ├── document_processor.py    # Text extraction & cleaning
│   ├── portfolio_manager.py     # Portfolio operations
│   └── portfolio_storage.py     # Portfolio S3 storage
├── llm/
│   ├── instructions.json        # LLM prompts & formats
│   └── llm_client.py           # LLM API client (TODO: implement)
└── templates/
    ├── custom.css               # UI styling
    └── custom.html              # HTML templates
```

## 🔑 AWS Permissions Required

Your IAM user/role needs:

- `s3:ListBucket` - List files in bucket
- `s3:GetObject` - Read files
- `s3:PutObject` - Upload files
- `s3:DeleteObject` - Delete/replace files

## 💡 Usage

### Document Management

1. **Upload Document**: Select file → Click "📤 Upload to S3" → Auto-loads
2. **View Document**: Select from dropdown → Click "Load Selected Document"
3. **Refresh List**: Click "🔄 Refresh Document List" after external changes

### Portfolio Management

1. **Create Portfolio**: Add stocks → Click "💾 Save Portfolio"
2. **Load Portfolio**: Select portfolio → Auto-extracts SEC filings
3. **SEC Extraction**: Automatically extracts 18 key sections per company
4. **View Extractions**: Check S3 at `extracted_filings/{portfolio}/`

### Analysis (when LLM implemented)

1. Load document → Click analysis buttons
2. Get regulatory impact scores
3. Receive portfolio recommendations

## 🛠️ Tech Stack

- **Frontend**: Gradio 4.0+
- **Storage**: AWS S3
- **Backend**: Python 3.8+, Boto3
- **Processing**: BeautifulSoup4, lxml, Pandas
- **SEC Parsing**: sec-parser 0.50+ (optional)
- **Financial Data**: yfinance
- **AI/LLM**: Perplexity/OpenAI API (to be configured)

## 📝 Data Structure in S3

```
s3://your-bucket/
├── data/
│   ├── directives/              # Regulatory documents (HTML/XML)
│   ├── fillings/                # Raw company 10-K filings
│   │   ├── AAPL/
│   │   │   └── 2024-11-01-10k-AAPL.html
│   │   ├── MSFT/
│   │   └── ...
│   ├── 2025-08-15_composition_sp500.csv
│   └── 2025-09-26_stocks-performance.csv
├── extracted_filings/           # Parsed SEC filing sections (NEW)
│   ├── {portfolio_name}/
│   │   ├── AAPL/
│   │   │   └── 10_K_20251102_120000_complete.json
│   │   ├── MSFT/
│   │   │   └── 10_K_20251102_120005_complete.json
│   │   └── ...
│   └── another_portfolio/
│       └── ...
└── {portfolio_id}/              # Portfolio CSV files
    └── portfolio_{name}.csv
```

### Extracted Filing JSON Structure

```json
{
  "metadata": {
    "ticker": "AAPL",
    "filing_type": "10-K",
    "extraction_date": "2025-11-02T12:00:00",
    "extractor_version": "2.1"
  },
  "sections": {
    "business": "...",
    "risk_factors": "...",
    "cybersecurity": "...",
    "properties": "...",
    "legal_proceedings": "...",
    "market_for_equity": "...",
    "mda": "...",
    "market_risk": "...",
    "financial_statements": "...",
    "controls_and_procedures": "...",
    "executive_compensation": "...",
    "material_agreements": "...",
    "cybersecurity_incidents": "...",
    "acquisitions_dispositions": "...",
    "earnings_announcements": "...",
    "material_impairments": "...",
    "accountant_changes": "...",
    "control_changes": "...",
    "officer_director_changes": "..."
  },
  "statistics": {
    "sections_found": [...],
    "total_sections_count": 12
  },
  "portfolio_impact_focus": {
    "high_impact_sections": [...],
    "risk_indicators": [...],
    "value_drivers": [...]
  }
}
```

## 🔧 Next Steps

1. **Implement LLM Integration** - Add API calls in `llm/llm_client.py`

   - Cross-reference SEC filings with regulatory documents
   - Generate portfolio recommendations based on regulatory changes
   - Assess regulatory impact scores per company

2. **Enhance Analysis Features**

   - Connect analysis buttons to LLM functions
   - Use extracted SEC sections for regulatory impact modeling
   - Compare risk factors with regulatory requirements

3. **Add Visualizations**

   - Charts for impact analysis
   - Portfolio risk heatmaps
   - Regulatory compliance dashboards

4. **Optimize Performance**
   - Add caching for processed documents
   - Parallel extraction for multiple companies
   - Background processing for large portfolios

## 📚 SEC Filing Extractor API

### Extract Filing Sections

```python
from utils.sec_filing_extractor import extract_filing_from_html

# Extract sections and save to S3
result = extract_filing_from_html(
    ticker="AAPL",
    html_content=html_content,
    portfolio_name="tech_portfolio",
    filing_type="10-K",
    save_to_s3=True
)

# Access extracted sections
business = result['business']
risk_factors = result['risk_factors']
legal_proceedings = result['legal_proceedings']
```

### Read Extracted Filings from S3

```python
from utils.s3_utils import read_file_from_s3, list_files_in_s3
import json

# List all filings for a portfolio
files = list_files_in_s3("extracted_filings/tech_portfolio/")

# Read a specific filing
s3_key = "extracted_filings/tech_portfolio/AAPL/10_K_20251102_120000_complete.json"
json_content = read_file_from_s3(s3_key)
data = json.loads(json_content)

# Access sections
sections = data['sections']
```

## 🐛 Troubleshooting

### S3 Connection Issues

- Verify AWS credentials in `.env` file
- Check that `AWS_SESSION_TOKEN` is set if using temporary credentials
- Test with: `aws s3 ls s3://your-bucket/`

### SEC Parser Not Available

- The tool works without `sec-parser` (uses BeautifulSoup only)
- For enhanced parsing, install: `pip install sec-parser`
- Optional dependency for semantic element extraction
