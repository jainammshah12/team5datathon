# ComplianceVision  🚀

**AI-Powered Portfolio Management Tool for Regulatory Risk Assessment**

An intelligent analysis and simulation platform that leverages generative AI and NLP to transform regulatory complexity into actionable portfolio management insights. Designed for institutional investors navigating an increasingly complex global regulatory landscape.

---

## 📖 Project Summary

In today's financial markets, institutional investors face mounting challenges from complex regulatory frameworks, protectionist policies, and economic sanctions. This tool addresses that challenge by automatically analyzing new regulatory texts, extracting key requirements, and assessing their financial impact on equity portfolios—specifically the S&P 500.

**Key Objectives:**

- **Automate Regulatory Analysis**: Use NLP and generative AI to extract entities, requirements, and deadlines from diverse regulatory documents (laws, directives, sanctions)
- **Assess Financial Impact**: Cross-reference regulations with company SEC filings (10-K/10-Q) to evaluate risks and opportunities at the individual stock, sector, and portfolio levels
- **Generate Strategic Recommendations**: Provide actionable portfolio adjustments (sector rotation, security replacement, weight reallocation) with quantified risk scores
- **Enable Proactive Decision-Making**: Transform regulatory monitoring from reactive compliance into proactive portfolio optimization

**Real-World Application:**

For example, a 100% tariff on semiconductors might not directly impact Apple's profits (phones manufactured in India/China), but tariffs on consumer electronics would pressure iPhone margins in the US market. This tool identifies such nuanced impacts by analyzing business models from SEC filings alongside regulatory requirements.

---

## ✨ Key Features

- **Document Management**: Upload regulatory documents (HTML, XML, TXT, Markdown), automatically stored and versioned in AWS S3. Multi-language, full-text extraction with instant analysis.
- **NLP Pipeline**: Two-layer system combining spaCy (entity/key phrase extraction) and OpenAI/Perplexity (impact and summary), with caching to avoid repeat processing.
- **SEC Filing Extraction**: Automated extraction of key sections from 10-K/10-Q filings, organized by portfolio, with metadata and smart cleanup.
- **AI-Powered Analysis**: Assesses regulatory and financial impact, highlights affected sectors/companies, and generates actionable, prioritized portfolio recommendations.
- **Portfolio Management**: Easily create/manage portfolios, with automatic SEC filing extraction, key metrics, and visual adjustment tracking.
- **Data Explorer**: Browse S&P 500 composition, stock performance, SEC filings, and extracted sections with interactive tables and export options.



---

## 🛠️ Technology Stack

### **Frontend & UI**
- **Gradio 4.0+**: Modern, responsive web interface with tabbed navigation
- **Custom CSS/HTML**: Professional styling with dark theme and enhanced UX
- **Interactive components**: Real-time updates, collapsible sections, and dynamic tables

### **Backend & Core Processing**
- **Python 3.8+**: Core application logic
- **Boto3**: AWS S3 integration for cloud storage
- **Pandas**: Data manipulation and portfolio analysis
- **Beautiful Soup 4 & lxml**: HTML/XML parsing for document extraction
- **sec-parser 0.50+**: Advanced SEC filing structure analysis (optional)

### **Natural Language Processing**
- **spaCy 3.6+**: Fast entity extraction and linguistic analysis
  - Model: `en_core_web_sm` (English language support)
  - Named entity recognition (NER)
  - Dependency parsing and noun phrase extraction
  
- **langdetect**: Automatic language detection for multilingual documents

### **AI & Machine Learning**
- **OpenAI API (GPT-4/GPT-4o-mini)**: Advanced language understanding and impact analysis
- **Perplexity AI API (sonar-pro/sonar)**: Real-time internet-enabled financial intelligence
  - Priority choice for current market data
  - Access to recent news and regulatory updates
  
- **Deep Translator**: Multi-language support for global regulations

### **Financial Data**
- **yfinance**: Real-time stock market data and company information
- **SEC EDGAR**: Direct access to 10-K, 10-Q, and 8-K filings

### **Cloud Infrastructure**
- **AWS S3**: Scalable object storage for documents, portfolios, and extracted data
- **IAM**: Secure credential management with temporary session tokens

### **Development & Operations**
- **python-dotenv**: Environment configuration management
- **tqdm**: Progress tracking for batch operations
- **requests**: HTTP client for API interactions
- **Plotly**: Interactive data visualizations (for future enhancements)

---

## 🏗️ Architecture

### **System Architecture & Data Flow**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GRADIO WEB INTERFACE                                │
│         [Documents] [Analysis] [Portfolio] [Data Explorer]                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        ┌───────────▼──────────┐  ┌──────────▼──────────┐
        │  1. DOCUMENT UPLOAD  │  │  2. PORTFOLIO SETUP │
        │  (Regulatory Texts)  │  │  (Ticker + Weights) │
        └──────────┬───────────┘  └──────────┬──────────┘
                   │                         │
                   │ Upload to S3            │ Save to S3 + Trigger
                   ▼                         ▼
        ┌──────────────────────────────────────────────────┐
        │            AWS S3 STORAGE LAYER                   │
        │  ┌──────────────┐  ┌────────────────────────┐    │
        │  │ Directives/  │  │ SEC Filings/ (by ticker)│   │
        │  │ Regulations  │  │ + Portfolios/          │    │
        │  └──────┬───────┘  └─────────┬──────────────┘    │
        └─────────┼──────────────────────┼──────────────────┘
                  │                      │
                  │                      │
        ┌─────────▼─────────────┐       │
        │  3. NLP PIPELINE      │       │
        │  (Directive Analysis) │       │
        ├───────────────────────┤       │
        │ LAYER 1: spaCy       │       │
        │ • Entity extraction   │       │
        │ • Key phrases         │       │
        │ • Sentiment analysis  │       │
        ├───────────────────────┤       │
        │ LAYER 2: LLM (OpenAI/ │       │
        │          Perplexity)  │       │
        │ • Affected sectors    │       │
        │ • Financial impact    │       │
        │ • Executive summary   │       │
        │ • Market rating       │       │
        └───────────┬───────────┘       │
                    │                   │
                    │ Cache to S3       │
                    ▼                   │
        ┌─────────────────────┐         │
        │ Extracted NLP Data  │         │
        │ (entities, impacts) │         │
        └─────────┬───────────┘         │
                  │                     │
                  │            ┌────────▼──────────────┐
                  │            │ 4. SEC FILING         │
                  │            │    EXTRACTION         │
                  │            ├───────────────────────┤
                  │            │ Extract 18 sections:  │
                  │            │ • Risk Factors        │
                  │            │ • Earnings            │
                  │            │ • Legal Proceedings   │
                  │            │ • Cybersecurity       │
                  │            │ • Material Impairments│
                  │            │ • MD&A, etc.          │
                  │            └────────┬──────────────┘
                  │                     │
                  │                     │ Save to S3
                  │                     ▼
                  │            ┌─────────────────────┐
                  │            │ Extracted Filings/  │
                  │            │ {portfolio}/{ticker}│
                  │            └────────┬────────────┘
                  │                     │
                  │                     │
                  └──────────┬──────────┘
                             │
        ┌────────────────────▼────────────────────────┐
        │  5. AI RECOMMENDATION ENGINE                │
        ├─────────────────────────────────────────────┤
        │  INPUTS:                                    │
        │  • Regulatory entities & requirements       │
        │  • Company SEC filing sections              │
        │  • Portfolio holdings & weights             │
        │                                             │
        │  PROCESSING:                                │
        │  • Cross-reference regulations with filings │
        │  • Assess compliance costs & impacts        │
        │  • Evaluate risk factors vs. regulations    │
        │  • Analyze sector exposure                  │
        │                                             │
        │  LLM CALL (Perplexity/OpenAI):             │
        │  • Evidence-based analysis                  │
        │  • Internet-enabled (Perplexity)           │
        │  • Multi-dimensional risk scoring           │
        └────────────┬────────────────────────────────┘
                     │
                     │ Generate
                     ▼
        ┌────────────────────────────────────┐
        │  6. PORTFOLIO RECOMMENDATIONS      │
        ├────────────────────────────────────┤
        │  For each holding:                 │
        │  • Action (Buy/Hold/Reduce/Sell)   │
        │  • Priority (Critical/High/Medium) │
        │  • Current vs. Recommended Weight  │
        │  • Rationale (with SEC citations)  │
        │                                    │
        │  Overall:                          │
        │  • Strategy summary                │
        │  • Risk assessment                 │
        │  • Alternative recommendations     │
        └────────────┬───────────────────────┘
                     │
                     │ Display
                     ▼
        ┌────────────────────────────────────┐
        │     USER INTERFACE (Analysis Tab)  │
        │  • Interactive recommendations     │
        │  • Sortable by priority            │
        │  • Expandable rationale            │
        │  • Export capabilities             │
        └────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                         │
├─────────────────────────────────────────────────────────────┤
│  • OpenAI API (gpt-4o-mini) - LLM without internet          │
│  • Perplexity API (sonar-pro) - LLM with internet access    │
│  • Yahoo Finance API - Real-time market data                │
│  • SEC EDGAR - Raw company filings                          │
└─────────────────────────────────────────────────────────────┘
```

**Key Data Flows:**

1. **Document Upload Flow**: User uploads regulatory document → Saved to S3 → NLP pipeline extracts entities/impacts → Cached for reuse
2. **Portfolio Setup Flow**: User creates portfolio → Saved to S3 → Triggers automatic SEC filing extraction → 18 sections per company extracted → Stored by portfolio/ticker
3. **Analysis Flow**: User selects directives + portfolio → System loads NLP results + SEC extractions → LLM analyzes cross-impacts → Generates prioritized recommendations
4. **Recommendation Flow**: Regulatory requirements + Company risk factors → AI engine assesses compliance costs/risks → Evidence-based buy/hold/sell actions with weight adjustments

### **Component Breakdown**

```
project_root/
├── gradio_app.py                 # Main application entry point
│
├── llm/                          # AI & Language Models
│   ├── llm_client.py            # OpenAI/Perplexity API wrapper
│   └── instructions.json        # Prompt templates & formats
│
├── utils/                        # Core utilities
│   ├── s3_utils.py              # S3 operations (CRUD)
│   ├── sec_filing_extractor.py  # 10-K/10-Q section parser
│   ├── document_processor.py    # HTML/XML text extraction
│   ├── directive_analyzer.py    # Regulatory document analysis
│   ├── directive_nlp_pipeline.py # Two-layer NLP processing
│   ├── portfolio_manager.py     # Portfolio CRUD operations
│   ├── portfolio_storage.py     # Portfolio S3 persistence
│   ├── filing_loader.py         # SEC filing batch loader
│   └── yfinance_fetcher.py      # Market data retrieval
│
├── templates/                    # UI customization
│   ├── custom.css               # Styling and theme
│   └── custom.html              # HTML templates
│
├── data/                         # Local data cache
│   ├── directives/              # Regulatory documents
│   ├── fillings/                # SEC filings (by ticker)
│   ├── 2025-08-15_composition_sp500.csv
│   └── 2025-09-26_stocks-performance.csv
│
├── requirements.txt              # Python dependencies
├── env.example                   # Environment configuration template
└── README.md                     # Documentation (this file)
```

### **Data Flow**

1. **Document Upload** → S3 Storage → Document Processor → NLP Pipeline → Extracted Entities
2. **Portfolio Creation** → Load Tickers → SEC Filing Loader → Section Extractor → S3 Storage
3. **Analysis Request** → Load Directive + Filings → LLM Client → Impact Assessment → Recommendations
4. **User Interaction** → Gradio Interface → Backend Processing → Real-time Updates

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- AWS S3 access
- Perplexity (recommended) or OpenAI API key

### 1. Clone & Install

```bash
git clone <repository-url>
cd team5datathon
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up AWS & .env

- Create an S3 bucket (e.g. `regulatory-analyzer-data`) in your AWS account.
- Ensure your IAM user has basic S3 read/write permissions.
- Copy `.env` template and add your credentials (AWS & LLM API keys):

```bash
cp env.example .env
```
- Only set `AWS_SESSION_TOKEN` if you use temporary keys (start with ASIA).
- Perplexity API is recommended for real-time data.

### 3. Upload Data

```bash
aws s3 cp data/2025-08-15_composition_sp500.csv s3://your-bucket-name/data/
aws s3 cp data/2025-09-26_stocks-performance.csv s3://your-bucket-name/data/
aws s3 sync data/directives/ s3://your-bucket-name/data/directives/
aws s3 sync data/fillings/ s3://your-bucket-name/data/fillings/
```
*(Or upload using the app after launch.)*

### 4. Run the App

```bash
python gradio_app.py
```
App runs at: http://localhost:7860

---


---

## 💡 Usage Guide

### Tab 1: 📄 Documents
- **Upload**: Click **"Browse Files"** and **"Upload to S3"** to add regulatory docs (`.html`, `.xml`, `.txt`, `.md`). Files are saved to S3 and appear instantly.
- **Load/View**: Select from dropdown and click **"Load"** to view preview of the text.

---

### Tab 2: 🔍 Analysis
- **Select & Load Directives**: Check up to 3 directives and click **"Load Selected"**.
- **NLP Analysis**: Click **"Process with NLP Pipeline"** for a two-layer AI review (entity/key phrase extraction, impact summarization, affected sectors, financial estimates).
- **Portfolio Recommendations**: (Portfolio required—see Tab 3.) Click **"Generate Portfolio Recommendations"** for buy/hold/sell/weight suggestions, ranked by priority and explained with SEC citations.  
- **Explore Company Details**: Click a ticker for deeper filing/risk info.

---

### Tab 3: 💼 Portfolio
- **Create**: Name your portfolio, add tickers and weights, then **"Add Stock"** and **"Save"**. Autosaves as CSV to S3, triggers SEC filing extraction.
- **Load**: Refresh portfolio list, select and load a portfolio—holdings and extraction progress display automatically.
- **View Extractions**: Rapid access to key SEC filing sections: risk factors, earnings, material events, legal actions, cyber disclosures, etc.
- **Adjust**: Update weights based on AI suggestions; save and re-run for updated recommendations.

---

### Tab 4: 📊 Data Explorer
- **S&P 500 Companies**: Browse/search all companies with sector, industry, and market cap data. Export to CSV if needed.
- **Stock Performance**: Sortable historical metrics—returns, volatility, dividends.
- **SEC Filings**: Select a ticker to view/download filings (10-K, 10-Q, 8-K).
- **Extracted Sections**: Pick a portfolio and ticker to quickly read focused sections like business description, risk factors, or download as JSON.



---

## 🔧 Troubleshooting

- **AWS S3 Issues**  
  - Make sure your `.env` has correct AWS keys.
  - Test S3 with: `aws s3 ls s3://your-bucket-name/`
  - For temporary session tokens, refresh and update `AWS_SESSION_TOKEN`.
  - Check IAM permissions and AWS region if you get "Access Denied".

- **LLM API Problems**  
  - Verify your Perplexity/OpenAI API keys in `.env`.  
  - Ensure your API key is still active at:  
    - [Perplexity](https://www.perplexity.ai/settings/api)  
    - [OpenAI](https://platform.openai.com/api-keys)
  - If you see 429 errors, wait a minute before retrying.

- **spaCy Model Error**  
  If you see a spaCy model error, run:  
  ```
  python -m spacy download en_core_web_sm
  ```

- **SEC Filing Extraction Fails**  
  - Check that required filings exist in S3.
  - Install `sec-parser` for best results:  
    `pip install sec-parser`
  - Tool supports HTML/XML, not PDF.

- **Portfolio/Gradio Issues**  
  - Portfolio files need columns: `ticker`, `weight` (as decimals, e.g. 0.10).
  - If interface won't load, check your Gradio version (`pip show gradio`), clear browser cache, and restart the app.

- **Performance Tips**  
  - Analyze fewer directives/stocks at once.
  - Take advantage of caching (repeated runs are faster).
  - For faster LLM, switch to sonar (`llm/llm_client.py` line 690).


---

## 📝 S3 Data Structure

All data is organized in S3 as follows:

```
s3://your-bucket-name/
│
├── data/
│   ├── directives/              # Regulatory documents
│   ├── filings/                 # Raw SEC filings (by ticker)
│   ├── ...csv                   # Other dataset files (S&P 500, performance)
│
├── extracted_filings/           # Processed SEC filings, by portfolio & ticker
├── data/extracted_directives/   # NLP outputs (by portfolio)
├── portfolios/                  # Portfolio CSVs
```

### **File Format Examples**

- **Extracted Filing JSON**: Contains metadata, extracted sections (e.g., business, risk_factors), and section statistics.
- **Portfolio CSV**: List of tickers, weights, sectors, and dates.
- **NLP Extraction JSON**: Metadata, Layer 1 (entities, phrases), Layer 2 (sector impact, summary).

(See `/data/` and `/extracted_filings/` for more samples.)


---

## 👥 Team 5 (Members)

- **Jainam Shah**
- **Hubert Lefebvre**
- **Jose Del Portillo Neira**
- **Bhavya Ruparelia**

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **S&P 500 Data**: S&P Dow Jones Indices LLC
- **SEC Filings**: U.S. Securities and Exchange Commission (EDGAR)
- **Regulatory Documents**: EU Official Journal, U.S. Congress, China Government, Japan Ministry
- **NLP Models**: spaCy, OpenAI, Perplexity AI
- **UI Framework**: Gradio by Hugging Face

---


## 🔮 Future Enhancements

- **Real-time alerts**: Notify on new regulatory changes
- **Portfolio backtesting**: Historical simulation of recommended adjustments
- **Multi-currency support**: International portfolio analysis
- **Advanced visualizations**: Interactive charts with Plotly
- **Collaboration features**: Multi-user portfolio sharing
- **Mobile responsiveness**: Optimized UI for tablets and phones

---

**Built with ❤️**
