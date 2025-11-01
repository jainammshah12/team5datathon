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

### 🔍 Analysis (Placeholder)
- Extract entities from regulatory documents
- Evaluate financial impact on S&P 500 companies
- Identify affected sectors and companies
- *Note: LLM integration pending - implement in `llm/llm_client.py`*

### 💼 Portfolio (Placeholder)
- Portfolio adjustment recommendations
- Financial impact simulations
- Risk assessment
- *Note: Requires LLM implementation*

### 📊 Data Explorer
- S&P 500 company composition
- Stock performance data
- All data from S3 bucket

## 🏗️ Architecture

```
gradio_app.py           # Main Gradio interface
├── utils/
│   ├── s3_utils.py     # S3 operations (read/write/delete)
│   └── document_processor.py  # Text extraction & cleaning
├── llm/
│   ├── instructions.json      # LLM prompts & formats
│   └── llm_client.py         # LLM API client (TODO: implement)
└── templates/
    ├── custom.css     # UI styling
    └── custom.html    # HTML templates
```

## 🔑 AWS Permissions Required

Your IAM user/role needs:
- `s3:ListBucket` - List files in bucket
- `s3:GetObject` - Read files
- `s3:PutObject` - Upload files
- `s3:DeleteObject` - Delete/replace files

## 💡 Usage

1. **Upload Document**: Select file → Click "📤 Upload to S3" → Auto-loads
2. **View Document**: Select from dropdown → Click "Load Selected Document"
3. **Refresh List**: Click "🔄 Refresh Document List" after external changes
4. **Analyze** (when LLM implemented): Load document → Click analysis buttons

## 🛠️ Tech Stack

- **Frontend**: Gradio 4.0+
- **Storage**: AWS S3
- **Backend**: Python 3.8+, Boto3
- **Processing**: BeautifulSoup4, Pandas
- **AI/LLM**: Perplexity/OpenAI API (to be configured)

## 📝 Data Structure in S3

```
s3://your-bucket/
├── data/
│   ├── directives/              # Regulatory documents
│   ├── fillings/                # Company 10-K filings
│   │   ├── AAPL/
│   │   ├── MSFT/
│   │   └── ...
│   ├── 2025-08-15_composition_sp500.csv
│   └── 2025-09-26_stocks-performance.csv
```

## 🔧 Next Steps

1. **Implement LLM Integration** - Add API calls in `llm/llm_client.py`
2. **Add Error Handling** - Improve validation and error messages
3. **Enhance Analysis** - Connect analysis buttons to LLM functions
4. **Add Caching** - Cache processed documents in S3
5. **Add Visualizations** - Charts for impact analysis
