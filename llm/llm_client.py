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

        # Create efficient prompt focusing on key risk indicators
        prompt_parts = [
            "Generate portfolio adjustment recommendations based on SEC 10-K filing data and regulatory directives.",
            "Analyze risks, financial health, material events, and regulatory compliance requirements for each company.",
            "\nPortfolio Holdings:",
        ]

        # Add regulatory directives context if available
        if directives_data:
            prompt_parts.append("\n=== Regulatory Directives ===")
            prompt_parts.append(
                f"Consider the following {len(directives_data)} regulatory directive(s) in your analysis:"
            )

            for directive_name, directive_info in list(directives_data.items())[
                :3
            ]:  # Limit to 3 directives to save tokens
                prompt_parts.append(f"\nDirective: {directive_name}")

                # Add key sections from directive
                if isinstance(directive_info, dict):
                    if directive_info.get("title"):
                        prompt_parts.append(f"Title: {directive_info['title'][:200]}")
                    if directive_info.get("effective_date"):
                        prompt_parts.append(
                            f"Effective Date: {directive_info['effective_date'][:200]}"
                        )
                    if directive_info.get("affected_sectors"):
                        prompt_parts.append(
                            f"Affected Sectors: {directive_info['affected_sectors'][:300]}"
                        )
                    if directive_info.get("financial_impacts"):
                        prompt_parts.append(
                            f"Financial Impacts: {directive_info['financial_impacts'][:800]}"
                        )
                    if directive_info.get("compliance_requirements"):
                        prompt_parts.append(
                            f"Compliance Requirements: {directive_info['compliance_requirements'][:800]}"
                        )
                    if directive_info.get("penalties_sanctions"):
                        prompt_parts.append(
                            f"Penalties/Sanctions: {directive_info['penalties_sanctions'][:600]}"
                        )

            prompt_parts.append(
                "\n⚠️ IMPORTANT: Factor regulatory compliance costs, penalties, and sector impacts when recommending portfolio adjustments."
            )

        # Add portfolio summary (compact)
        for ticker in portfolio_tickers:
            weight_pct = portfolio_weights.get(ticker, 0) * 100
            prompt_parts.append(f"- {ticker}: {weight_pct:.1f}%")

        prompt_parts.append("\nKey Filing Data (focus on risks and material events):")

        # Add only critical sections for each ticker (token optimization)
        for ticker in portfolio_tickers:
            if ticker in filing_sections:
                sections = filing_sections[ticker]
                prompt_parts.append(f"\n{ticker}:")

                # Only include sections with content
                if sections.get("risk_factors"):
                    prompt_parts.append(
                        f"Risk Factors: {sections['risk_factors'][:1500]}..."
                    )
                if sections.get("legal_proceedings"):
                    prompt_parts.append(
                        f"Legal Issues: {sections['legal_proceedings'][:800]}..."
                    )
                if sections.get("cybersecurity_incidents"):
                    prompt_parts.append(
                        f"Cybersecurity: {sections['cybersecurity_incidents'][:800]}..."
                    )
                if sections.get("material_impairments"):
                    prompt_parts.append(
                        f"Material Impairments: {sections['material_impairments'][:800]}..."
                    )
                if sections.get("earnings_announcements"):
                    prompt_parts.append(
                        f"Earnings: {sections['earnings_announcements'][:800]}..."
                    )

        prompt_parts.append("\nIMPORTANT GUIDELINES:")
        prompt_parts.append(
            "- When recommending to DECREASE exposure to a sector, ALWAYS suggest alternative sectors with specific tickers"
        )
        prompt_parts.append(
            "- Explain how alternative sectors complement the current portfolio (diversification benefits, risk reduction, growth opportunities)"
        )
        prompt_parts.append(
            "- Cash should ONLY be recommended if there are truly NO better investment options - prefer sector diversification"
        )
        prompt_parts.append(
            "- All recommended weights should sum to approximately 100%"
        )
        prompt_parts.append(
            "- If suggesting sector diversification, include it in the 'sector_diversification' field"
        )

        prompt_parts.append("\nProvide recommendations in JSON format:")
        prompt_parts.append(
            json.dumps(
                INSTRUCTIONS["portfolio_recommendation"]["output_format"], indent=2
            )
        )

        prompt = "\n".join(prompt_parts)

        # Check cache first
        cache_key = f"rec_{hash(prompt)}_{hash(json.dumps(sorted(portfolio_tickers)))}"
        if cache_key in _response_cache:
            print("[INFO] Using cached recommendation response")
            return _response_cache[cache_key]

        try:
            # Make API call (implementation depends on provider)
            response = self._call_llm_api(prompt, max_tokens=2500)

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

        Args:
            prompt: Prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            LLM response text
        """
        if not self.api_key:
            raise ValueError("No API key configured")

        # Try OpenAI first (if both keys exist, prefer OpenAI)
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
                print(f"[WARNING] OpenAI API failed: {e}")
                # Continue to try Perplexity if OpenAI fails

        # Try Perplexity if OpenAI not available or failed
        perplexity_key = os.getenv("PERPLEXITY_API_KEY") or (
            self.api_key if self.api_key and self.api_key.startswith("pplx-") else None
        )
        if not perplexity_key:
            raise Exception(
                "No valid API key found. Please configure OPENAI_API_KEY or PERPLEXITY_API_KEY in .env file."
            )
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {perplexity_key}",
                "Content-Type": "application/json",
            }

            # Use correct Perplexity model names (sonar for chat, sonar-pro for premium)
            # Try sonar-pro first, fallback to sonar
            perplexity_model = "sonar-pro"  # Premium model with better quality
            if not perplexity_key.startswith("pplx-"):
                # If API key doesn't look like Perplexity format, try sonar
                perplexity_model = "sonar"

            data = {
                "model": perplexity_model,
                "messages": [
                    {
                        "role": "system",
                        "content": INSTRUCTIONS.get("system_prompt", ""),
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
                timeout=60,
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
                    timeout=60,
                )

                if response.status_code != 200:
                    raise Exception(
                        f"Perplexity API error ({response.status_code}): {error_detail}"
                    )

            return response.json()["choices"][0]["message"]["content"]

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
            print(f"[ERROR] Perplexity API failed: {e}")
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
