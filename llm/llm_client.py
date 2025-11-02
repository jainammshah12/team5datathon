"""LLM client for document analysis and financial impact evaluation."""

import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import time
import re

load_dotenv()

# Import instructions
INSTRUCTIONS_PATH = os.path.join(os.path.dirname(__file__), "instructions.json")

with open(INSTRUCTIONS_PATH, "r") as f:
    INSTRUCTIONS = json.load(f)

# Cache for API responses to avoid redundant calls
_response_cache = {}


class LLMClient:
    """Client for interacting with LLM services."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the LLM service (defaults to env variable)
            model: Model name to use
        """
        self.api_key = (
            api_key or os.getenv("PERPLEXITY_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.system_prompt = INSTRUCTIONS.get("system_prompt", "")

    def extract_entities(self, document_text: str) -> Dict:
        """
        Extract entities and key information from a regulatory document.

        Args:
            document_text: The text content of the regulatory document

        Returns:
            Dictionary with extracted entities
        """
        if not self.api_key:
            print("[WARNING] No API key configured for entity extraction")
            return {
                "document_type": "Regulation",
                "jurisdiction": "Unknown",
                "effective_date": None,
                "key_requirements": ["API key required for detailed analysis"],
                "affected_sectors": [],
                "monetary_impacts": [],
                "deadlines": [],
                "mentioned_entities": [],
            }

        prompt = f"""{INSTRUCTIONS['entity_extraction']['task']}

Document (first 5000 chars):
{document_text[:5000]}

Extract entities in the following JSON format:
{json.dumps(INSTRUCTIONS['entity_extraction']['output_format'], indent=2)}

Focus on:
1. Document type and jurisdiction
2. Key requirements and affected sectors
3. Monetary impacts and deadlines
4. Mentioned companies/entities
"""

        cache_key = f"entities_{hash(document_text[:1000])}"
        if cache_key in _response_cache:
            return _response_cache[cache_key]

        try:
            response = self._call_llm_api(prompt, max_tokens=1500)
            entities = self._parse_json_response(
                response,
                {
                    "document_type": "Regulation",
                    "jurisdiction": "Unknown",
                    "effective_date": None,
                    "key_requirements": [],
                    "affected_sectors": [],
                    "monetary_impacts": [],
                    "deadlines": [],
                    "mentioned_entities": [],
                },
            )

            _response_cache[cache_key] = entities
            return entities

        except Exception as e:
            print(f"[ERROR] Entity extraction failed: {e}")
            return {
                "document_type": "Regulation",
                "jurisdiction": "Unknown",
                "effective_date": None,
                "key_requirements": [],
                "affected_sectors": [],
                "monetary_impacts": [],
                "deadlines": [],
                "mentioned_entities": [],
            }

    def analyze_impact(self, entities: Dict, companies_data: Dict) -> Dict:
        """
        Analyze financial impact on S&P 500 companies.

        Args:
            entities: Extracted entities from document
            companies_data: S&P 500 companies data

        Returns:
            Dictionary with impact analysis
        """
        if not self.api_key:
            print("[WARNING] No API key configured for impact analysis")
            return {
                "affected_companies": [],
                "sector_summary": {
                    "most_affected_sector": "Unknown",
                    "overall_impact": "Configure API key for detailed analysis",
                },
            }

        # Prepare company data summary
        sectors = {}
        for company in companies_data:
            sector = company.get("GICS Sector", "Unknown")
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(
                {
                    "ticker": company.get("Symbol", "N/A"),
                    "name": company.get("Security", "N/A"),
                    "sub_industry": company.get("GICS Sub-Industry", "N/A"),
                }
            )

        prompt = f"""{INSTRUCTIONS['impact_analysis']['task']}

Regulatory Document Entities:
{json.dumps(entities, indent=2)}

Available Sectors and Sample Companies:
{json.dumps({k: v[:5] for k, v in sectors.items()}, indent=2)}

Provide impact analysis in JSON format:
{json.dumps(INSTRUCTIONS['impact_analysis']['output_format'], indent=2)}

Focus on:
1. Which sectors are most affected
2. Top 10 most impacted companies with rationale
3. Impact level (high/medium/low) and type (positive/negative/neutral)
"""

        cache_key = f"impact_{hash(json.dumps(entities))}_{hash(json.dumps(sectors))}"
        if cache_key in _response_cache:
            return _response_cache[cache_key]

        try:
            response = self._call_llm_api(prompt, max_tokens=2000)
            impact = self._parse_json_response(
                response,
                {
                    "affected_companies": [],
                    "sector_summary": {
                        "most_affected_sector": "Unknown",
                        "overall_impact": "Pending analysis",
                    },
                },
            )

            _response_cache[cache_key] = impact
            return impact

        except Exception as e:
            print(f"[ERROR] Impact analysis failed: {e}")
            return {
                "affected_companies": [],
                "sector_summary": {
                    "most_affected_sector": "Unknown",
                    "overall_impact": "Pending analysis",
                },
            }

    def generate_recommendations(self, impact_analysis: Dict, portfolio: Dict) -> Dict:
        """
        Generate portfolio adjustment recommendations.

        Args:
            impact_analysis: Results from impact analysis
            portfolio: Current portfolio holdings

        Returns:
            Dictionary with recommendations
        """
        if not self.api_key:
            print("[WARNING] No API key configured for recommendations")
            holdings = portfolio.get("holdings", [])
            return {
                "recommendations": [
                    {
                        "action": "monitor",
                        "ticker": h.get("ticker", "N/A"),
                        "current_weight": h.get("weight", 0),
                        "recommended_weight": h.get("weight", 0),
                        "reason": "API key required for AI analysis",
                        "priority": "low",
                    }
                    for h in holdings
                ],
                "overall_strategy": "Configure LLM API key for detailed recommendations",
                "risk_assessment": "Manual review required",
            }

        holdings_summary = []
        for holding in portfolio.get("holdings", []):
            holdings_summary.append(
                {
                    "ticker": holding.get("ticker", "N/A"),
                    "weight": holding.get("weight", 0),
                }
            )

        prompt = f"""{INSTRUCTIONS['portfolio_recommendation']['task']}

Impact Analysis:
{json.dumps(impact_analysis, indent=2)}

Current Portfolio Holdings:
{json.dumps(holdings_summary, indent=2)}

Provide recommendations in JSON format:
{json.dumps(INSTRUCTIONS['portfolio_recommendation']['output_format'], indent=2)}

Consider:
1. Which holdings are most affected by the regulation
2. Recommended actions (buy/sell/hold/reduce)
3. Priority levels (critical/high/medium/low)
4. Suggested weight adjustments
"""

        try:
            response = self._call_llm_api(prompt, max_tokens=2000)
            recommendations = self._parse_json_response(
                response,
                {
                    "recommendations": [],
                    "overall_strategy": "Hold current positions pending further analysis",
                    "risk_assessment": "Moderate risk",
                },
            )

            if recommendations.get("recommendations"):
                recommendations["recommendations"] = sorted(
                    recommendations["recommendations"],
                    key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                        x.get("priority", "low").lower(), 3
                    ),
                )

            return recommendations

        except Exception as e:
            print(f"[ERROR] Recommendation generation failed: {e}")
            holdings = portfolio.get("holdings", [])
            return {
                "recommendations": [
                    {
                        "action": "monitor",
                        "ticker": h.get("ticker", "N/A"),
                        "current_weight": h.get("weight", 0),
                        "recommended_weight": h.get("weight", 0),
                        "reason": "Analysis failed - please try again",
                        "priority": "low",
                    }
                    for h in holdings
                ],
                "overall_strategy": "Hold current positions pending further analysis",
                "risk_assessment": "Moderate risk",
            }

    def generate_portfolio_recommendations_from_filings(
        self,
        portfolio_tickers: List[str],
        portfolio_weights: Dict[str, float],
        filing_sections: Dict[str, Dict[str, str]],
        directives_data: Dict[str, Dict[str, str]] = None,
    ) -> Dict:
        """
        Generate portfolio recommendations based on SEC filing sections.
        Optimized to minimize token usage by using only relevant sections.

        Args:
            portfolio_tickers: List of ticker symbols in portfolio
            portfolio_weights: Dictionary mapping ticker -> weight (0-1)
            filing_sections: Dictionary mapping ticker -> relevant filing sections

        Returns:
            Dictionary with recommendations sorted by priority
        """
        # Check if API key is configured (check both instance and environment)
        api_key = (
            self.api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("PERPLEXITY_API_KEY")
        )
        if not api_key:
            print("[WARNING] No LLM API key found. Using fallback recommendations.")
            print(
                "[INFO] To enable AI recommendations, add OPENAI_API_KEY or PERPLEXITY_API_KEY to .env file"
            )
            return self._get_fallback_recommendations(
                portfolio_tickers, portfolio_weights
            )

        # Use the API key if we found one in environment but not in instance
        if not self.api_key:
            self.api_key = api_key

        # Create comprehensive prompt with detailed analysis requirements
        prompt_parts = [
            "=== PORTFOLIO ANALYSIS & RECOMMENDATIONS REQUEST ===",
            "",
            "OBJECTIVE: Generate comprehensive, evidence-based portfolio adjustment recommendations by analyzing SEC filing data and regulatory directives. Provide actionable insights that balance risk-adjusted returns with regulatory compliance.",
            "",
            "ANALYSIS REQUIREMENTS:",
            "1. FINANCIAL HEALTH: Evaluate each company's financial condition from SEC filings (revenue trends, profit margins, debt levels, cash flow, material impairments)",
            "2. RISK ASSESSMENT: Identify specific risks from filing sections (operational, financial, regulatory, legal, cybersecurity)",
            "3. REGULATORY IMPACT: Assess how selected regulatory directives affect each company (compliance costs, operational restrictions, competitive positioning)",
            "4. SECTOR DYNAMICS: Consider industry trends, competitive landscape, and macroeconomic factors",
            "5. PORTFOLIO OPTIMIZATION: Recommend weight adjustments that improve risk-adjusted returns and diversification",
            "",
            "=== CURRENT PORTFOLIO ===",
        ]

        # Add portfolio summary with weights
        prompt_parts.append("\nPortfolio Holdings:")
        total_holdings = len(portfolio_tickers)
        prompt_parts.append(f"Total Positions: {total_holdings}")
        
        for ticker in portfolio_tickers:
            weight_pct = portfolio_weights.get(ticker, 0) * 100
            prompt_parts.append(f"  • {ticker}: {weight_pct:.1f}% allocation")

        # Add regulatory directives context if available
        if directives_data:
            prompt_parts.append("")
            prompt_parts.append("=== REGULATORY DIRECTIVES ANALYSIS ===")
            prompt_parts.append(
                f"The following {len(directives_data)} regulatory directive(s) must be considered in the analysis:"
            )
            prompt_parts.append(
                "Evaluate: (1) Which portfolio companies are affected, (2) Magnitude of compliance costs, (3) Operational/strategic impacts, (4) Competitive advantages/disadvantages created"
            )

            for directive_name, directive_info in list(directives_data.items())[
                :3
            ]:  # Limit to 3 directives to save tokens
                prompt_parts.append("")
                prompt_parts.append(f"DIRECTIVE: {directive_name}")
                prompt_parts.append("-" * 80)

                # Add key sections from directive with better formatting
                if isinstance(directive_info, dict):
                    if directive_info.get("title"):
                        prompt_parts.append(f"📋 Title: {directive_info['title'][:250]}")
                    if directive_info.get("effective_date"):
                        prompt_parts.append(
                            f"📅 Effective Date: {directive_info['effective_date'][:200]}"
                        )
                    if directive_info.get("affected_sectors"):
                        prompt_parts.append(
                            f"🏢 Affected Sectors: {directive_info['affected_sectors'][:400]}"
                        )
                    if directive_info.get("financial_impacts"):
                        prompt_parts.append(
                            f"💰 Financial Impacts: {directive_info['financial_impacts'][:1000]}"
                        )
                    if directive_info.get("compliance_requirements"):
                        prompt_parts.append(
                            f"✅ Compliance Requirements: {directive_info['compliance_requirements'][:1000]}"
                        )
                    if directive_info.get("penalties_sanctions"):
                        prompt_parts.append(
                            f"⚠️ Penalties/Sanctions: {directive_info['penalties_sanctions'][:800]}"
                        )

            prompt_parts.append("")
            prompt_parts.append(
                "🎯 DIRECTIVE ANALYSIS TASK: Assess which portfolio companies face the highest regulatory burden, estimated compliance costs (% of revenue/margin impact), and whether any companies gain competitive advantages. Recommend portfolio adjustments accordingly."
            )

        prompt_parts.append("")
        prompt_parts.append("=== SEC FILING DATA & RISK ANALYSIS ===")
        prompt_parts.append("Key sections from latest 10-K/Q filings (focus on material risks and financial health):")

        # Add comprehensive filing data for each ticker
        for ticker in portfolio_tickers:
            if ticker in filing_sections:
                sections = filing_sections[ticker]
                weight_pct = portfolio_weights.get(ticker, 0) * 100
                
                prompt_parts.append("")
                prompt_parts.append(f"{'='*80}")
                prompt_parts.append(f"COMPANY: {ticker} (Current Allocation: {weight_pct:.1f}%)")
                prompt_parts.append(f"{'='*80}")

                sections_found = []
                
                # Risk Factors - most critical for investment decisions
                if sections.get("risk_factors"):
                    prompt_parts.append("")
                    prompt_parts.append("🚨 RISK FACTORS:")
                    prompt_parts.append(sections['risk_factors'][:2000])
                    if len(sections['risk_factors']) > 2000:
                        prompt_parts.append("... [truncated for length]")
                    sections_found.append("risk_factors")

                # Financial Performance & Earnings
                if sections.get("earnings_announcements"):
                    prompt_parts.append("")
                    prompt_parts.append("📊 FINANCIAL PERFORMANCE:")
                    prompt_parts.append(sections['earnings_announcements'][:1200])
                    if len(sections['earnings_announcements']) > 1200:
                        prompt_parts.append("... [truncated for length]")
                    sections_found.append("earnings")
                
                # Material Impairments - indicates financial distress
                if sections.get("material_impairments"):
                    prompt_parts.append("")
                    prompt_parts.append("💸 MATERIAL IMPAIRMENTS & WRITE-DOWNS:")
                    prompt_parts.append(sections['material_impairments'][:1000])
                    if len(sections['material_impairments']) > 1000:
                        prompt_parts.append("... [truncated for length]")
                    sections_found.append("impairments")
                    
                # Legal Proceedings - potential liabilities
                if sections.get("legal_proceedings"):
                    prompt_parts.append("")
                    prompt_parts.append("⚖️ LEGAL PROCEEDINGS & REGULATORY ACTIONS:")
                    prompt_parts.append(sections['legal_proceedings'][:1000])
                    if len(sections['legal_proceedings']) > 1000:
                        prompt_parts.append("... [truncated for length]")
                    sections_found.append("legal")
                    
                # Cybersecurity - increasingly important risk factor
                if sections.get("cybersecurity_incidents"):
                    prompt_parts.append("")
                    prompt_parts.append("🔒 CYBERSECURITY RISKS & INCIDENTS:")
                    prompt_parts.append(sections['cybersecurity_incidents'][:1000])
                    if len(sections['cybersecurity_incidents']) > 1000:
                        prompt_parts.append("... [truncated for length]")
                    sections_found.append("cybersecurity")
                
                # Management Discussion & Analysis (if available)
                if sections.get("management_discussion"):
                    prompt_parts.append("")
                    prompt_parts.append("💼 MANAGEMENT DISCUSSION & ANALYSIS:")
                    prompt_parts.append(sections['management_discussion'][:1000])
                    if len(sections['management_discussion']) > 1000:
                        prompt_parts.append("... [truncated for length]")
                    sections_found.append("md&a")
                
                if not sections_found:
                    prompt_parts.append("⚠️ No detailed filing sections available for this ticker")
                else:
                    prompt_parts.append("")
                    prompt_parts.append(f"✓ Available data: {', '.join(sections_found)}")

        prompt_parts.append("")
        prompt_parts.append("="*80)
        prompt_parts.append("=== ANALYSIS GUIDELINES & REQUIREMENTS ===")
        prompt_parts.append("="*80)
        prompt_parts.append("")
        
        prompt_parts.append("📋 EVIDENCE-BASED ANALYSIS:")
        prompt_parts.append("  • Cite specific evidence from SEC filings or regulatory documents for EVERY recommendation")
        prompt_parts.append("  • Reference actual text from risk factors, financial metrics, or regulatory requirements")
        prompt_parts.append("  • Quantify impacts when possible (e.g., '10-15% revenue at risk', '3-5% margin compression')")
        prompt_parts.append("")
        
        prompt_parts.append("⚖️ PORTFOLIO CONSTRUCTION RULES:")
        prompt_parts.append("  • All recommended weights MUST sum to 100%")
        prompt_parts.append("  • Provide specific numeric targets (current weight → recommended weight)")
        prompt_parts.append("  • Ensure adequate diversification across sectors, market caps, and risk profiles")
        prompt_parts.append("  • Consider portfolio correlation structure and concentration risk")
        prompt_parts.append("")
        
        prompt_parts.append("🔄 ALTERNATIVE INVESTMENTS:")
        prompt_parts.append("  • When recommending to DECREASE exposure, ALWAYS suggest specific alternatives with tickers")
        prompt_parts.append("  • Explain how alternatives complement existing holdings (correlation, diversification, growth)")
        prompt_parts.append("  • Prefer sector rotation over cash allocation unless market conditions clearly warrant defensive positioning")
        prompt_parts.append("  • Include allocation percentages for recommended alternative sectors")
        prompt_parts.append("")
        
        prompt_parts.append("📊 RISK & RETURN ASSESSMENT:")
        prompt_parts.append("  • Evaluate both upside potential AND downside risk for each recommendation")
        prompt_parts.append("  • Provide multi-dimensional risk analysis (regulatory, operational, financial, market)")
        prompt_parts.append("  • Assign confidence levels (high/medium/low) based on data quality and analysis certainty")
        prompt_parts.append("  • Estimate expected portfolio impact (return improvement, volatility reduction)")
        prompt_parts.append("")
        
        prompt_parts.append("⏱️ IMPLEMENTATION GUIDANCE:")
        prompt_parts.append("  • Specify timeframe for each action (immediate, 1-3 months, 3-6 months, long-term)")
        prompt_parts.append("  • Prioritize recommendations (critical/high/medium/low)")
        prompt_parts.append("  • Consider implementation practicality (liquidity, transaction costs, tax implications)")
        prompt_parts.append("  • Provide phased roadmap if major portfolio restructuring is needed")
        prompt_parts.append("")
        
        prompt_parts.append("🎯 REGULATORY COMPLIANCE:")
        prompt_parts.append("  • Assess which companies face highest regulatory burden from selected directives")
        prompt_parts.append("  • Estimate compliance costs as % of revenue or margin impact")
        prompt_parts.append("  • Identify companies with competitive advantages due to regulatory changes")
        prompt_parts.append("  • Factor in penalties, operational restrictions, and market access limitations")
        prompt_parts.append("")
        
        prompt_parts.append("✅ OUTPUT REQUIREMENTS:")
        prompt_parts.append("  • Use the exact JSON format specified below")
        prompt_parts.append("  • Include ALL required fields for each recommendation")
        prompt_parts.append("  • Provide comprehensive 'overall_strategy' and 'risk_assessment' summaries")
        prompt_parts.append("  • Fill out 'portfolio_metrics', 'implementation_roadmap', and 'top_opportunities/risks' sections")
        prompt_parts.append("  • Be specific, quantitative, and actionable in all recommendations")
        prompt_parts.append("")
        prompt_parts.append("="*80)
        prompt_parts.append("")

        prompt_parts.append("REQUIRED OUTPUT FORMAT (JSON):")
        prompt_parts.append(
            json.dumps(
                INSTRUCTIONS["portfolio_recommendation"]["output_format"], indent=2
            )
        )
        
        prompt_parts.append("")
        prompt_parts.append("⚠️ CRITICAL: Respond ONLY with valid JSON matching the format above. Include all fields.")
        prompt_parts.append("Begin your response with { and end with }")

        prompt = "\n".join(prompt_parts)

        # Check cache first
        cache_key = f"rec_{hash(prompt)}_{hash(json.dumps(sorted(portfolio_tickers)))}"
        if cache_key in _response_cache:
            print("[INFO] Using cached recommendation response")
            return _response_cache[cache_key]

        try:
            # Make API call with increased token limit for comprehensive analysis
            response = self._call_llm_api(prompt, max_tokens=4000)

            # Parse response
            recommendations = self._parse_json_response(
                response,
                self._get_fallback_recommendations(
                    portfolio_tickers, portfolio_weights
                ),
            )

            # Sort by priority: critical -> medium -> low
            recommendations["recommendations"] = sorted(
                recommendations.get("recommendations", []),
                key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    x.get("priority", "low").lower(), 3
                ),
            )

            # Cache response
            _response_cache[cache_key] = recommendations
            return recommendations

        except Exception as e:
            print(f"[ERROR] LLM API call failed: {e}")
            import traceback

            traceback.print_exc()
            return self._get_fallback_recommendations(
                portfolio_tickers, portfolio_weights
            )

    def _call_llm_api(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Make actual LLM API call. Supports OpenAI and Perplexity.
        Prioritizes Perplexity for internet-enabled real-time analysis.

        Args:
            prompt: Prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        if not self.api_key:
            raise ValueError("No API key configured")

        # Check which API keys are available
        openai_key = os.getenv("OPENAI_API_KEY") or (
            self.api_key if self.api_key and self.api_key.startswith("sk-") else None
        )
        perplexity_key = os.getenv("PERPLEXITY_API_KEY") or (
            self.api_key if self.api_key and self.api_key.startswith("pplx-") else None
        )
        
        # Prioritize Perplexity for internet access (better for real-time financial data)
        if perplexity_key:
            try:
                return self._call_perplexity_api(prompt, max_tokens, perplexity_key)
            except Exception as e:
                print(f"[WARNING] Perplexity API failed: {e}")
                # Fall back to OpenAI if available
                if openai_key:
                    print("[INFO] Falling back to OpenAI API...")
                else:
                    raise
        
        # Try OpenAI as fallback
        openai_key = os.getenv("OPENAI_API_KEY") or (
            self.api_key if self.api_key and self.api_key.startswith("sk-") else None
        )
        if openai_key:
            try:
                try:
                    import openai
                except ImportError:
                    print(
                        "[WARNING] openai package not installed. Install with: pip install openai"
                    )
                    raise ImportError("openai package required")

                client = openai.OpenAI(api_key=openai_key)
                
                # Note: Standard OpenAI models don't have internet access
                # For internet access, consider using Perplexity API instead
                print("[INFO] Using OpenAI API (no internet access)")
                response = client.chat.completions.create(
                    model=self.model or "gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": INSTRUCTIONS.get("system_prompt", ""),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[ERROR] OpenAI API failed: {e}")
                raise

        # If no API key available
        raise Exception(
            "No valid API key found. Please configure PERPLEXITY_API_KEY (recommended for internet access) or OPENAI_API_KEY in .env file."
        )

    def _call_perplexity_api(self, prompt: str, max_tokens: int, api_key: str) -> str:
        """
        Call Perplexity API with internet access enabled.
        Perplexity's sonar models have built-in internet access for real-time data.
        
        Args:
            prompt: Prompt to send
            max_tokens: Maximum tokens in response
            api_key: Perplexity API key
            
        Returns:
            LLM response text
        """
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Use sonar-pro for best quality with internet access
            # Perplexity's sonar models have real-time internet search capabilities
            perplexity_model = "sonar-pro"  # Premium model with internet access
            
            print(f"[INFO] Using Perplexity API with internet access (model: {perplexity_model})")

            # Perplexity's sonar models automatically have internet access
            # No need for extra parameters - they search by default
            data = {
                "model": perplexity_model,
                "messages": [
                    {
                        "role": "system",
                        "content": INSTRUCTIONS.get("system_prompt", "") + 
                                   "\n\nIMPORTANT: You have access to real-time internet data. Use this to provide up-to-date financial information, current market conditions, recent news, and regulatory updates when relevant to the analysis. Prioritize information from reliable financial sources like SEC.gov, Bloomberg, Reuters, and Yahoo Finance.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=90,  # Increased timeout for internet search
            )

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get(
                        "message", error_detail
                    )
                except:
                    pass

                # Try fallback model
                print(f"[INFO] Trying fallback model 'sonar' instead of 'sonar-pro'...")
                data["model"] = "sonar"
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=90,
                )

                if response.status_code != 200:
                    raise Exception(
                        f"Perplexity API error ({response.status_code}): {error_detail}"
                    )

            result = response.json()
            
            # Log if citations are available (Perplexity sometimes includes them)
            if result.get("citations"):
                print(f"[INFO] Perplexity used {len(result['citations'])} internet sources")
                for i, citation in enumerate(result.get("citations", [])[:3], 1):
                    print(f"  [{i}] {citation}")
            else:
                print(f"[INFO] Perplexity response generated (internet search enabled)")
            
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("error", {}).get(
                        "message", e.response.text
                    )
                    error_msg = f"Perplexity API error ({e.response.status_code}): {error_detail}"
                except:
                    error_msg = f"Perplexity API error ({e.response.status_code}): {e.response.text}"
            print(f"[ERROR] Perplexity API failed: {error_msg}")
            raise Exception(
                f"Perplexity API failed. Make sure your PERPLEXITY_API_KEY is valid. Error: {error_msg}"
            )
        except Exception as e:
            print(f"[ERROR] Perplexity API call failed: {e}")
            raise

    def _parse_json_response(self, response_text: str, fallback: Dict) -> Dict:
        """Parse JSON from LLM response with multiple fallback strategies."""
        # Try direct parse
        try:
            return json.loads(response_text)
        except:
            pass

        # Try extracting JSON with regex
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Try finding JSON in code blocks
        try:
            code_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if code_match:
                return json.loads(code_match.group(1))
        except:
            pass

        print(f"[WARNING] Could not parse JSON from response, using fallback")
        return fallback

    def _get_fallback_recommendations(
        self, tickers: List[str], portfolio_weights: Dict[str, float] = None
    ) -> Dict:
        """
        Return fallback recommendations when API is not available.
        Provides basic recommendations based on filing data availability.

        Args:
            tickers: List of ticker symbols
            portfolio_weights: Dictionary mapping ticker -> weight (0-1)
        """
        if portfolio_weights is None:
            portfolio_weights = {}

        return {
            "recommendations": [
                {
                    "action": "monitor",
                    "ticker": t,
                    "current_weight": portfolio_weights.get(t, 0),
                    "recommended_weight": portfolio_weights.get(t, 0),
                    "reason": "LLM API not configured. Please add OPENAI_API_KEY or PERPLEXITY_API_KEY to .env file for AI-powered recommendations based on SEC filing analysis.",
                    "priority": "low",
                }
                for t in tickers
            ],
            "overall_strategy": "Hold current positions. Configure LLM API key (OPENAI_API_KEY or PERPLEXITY_API_KEY) in .env file to enable AI-powered portfolio recommendations based on SEC filing analysis.",
            "risk_assessment": "Cannot assess automatically - LLM API not configured. Review SEC filings manually in the Data Explorer tab for detailed risk assessment.",
        }

    def run_simulation(self, portfolio: Dict, scenarios: List[Dict]) -> Dict:
        """
        Run portfolio simulation with different scenarios.

        Args:
            portfolio: Current portfolio holdings
            scenarios: List of scenario configurations

        Returns:
            Dictionary with simulation results
        """
        # TODO: Implement actual simulation logic
        return {
            "scenarios": [
                {
                    "name": "Baseline",
                    "expected_return": "0%",
                    "risk_metrics": {"volatility": "0%", "sharpe_ratio": 0},
                    "portfolio_value_change": "0%",
                }
            ]
        }

    def call_api(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Make API call to LLM service.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        # TODO: Implement actual API call
        # Example for Perplexity API or OpenAI API
        # if self.api_key:
        #     response = api_client.complete(prompt, model=self.model, max_tokens=max_tokens)
        #     return response

        return "LLM API call not yet implemented. Please configure your API key."


def get_llm_client() -> LLMClient:
    """Get initialized LLM client instance."""
    return LLMClient()
