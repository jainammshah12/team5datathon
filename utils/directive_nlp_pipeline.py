"""
Lazy-Processing Two-Layer NLP Pipeline for Regulatory Directives

This module implements an intelligent NLP pipeline that:
1. Checks if extraction already exists in S3 before processing
2. Layer 1: AWS Comprehend (or spaCy fallback) for entity/key phrase extraction
3. Layer 2: AWS Bedrock (Claude/Titan) for impact summarization

The pipeline avoids redundant processing by reusing cached results.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import S3 utilities
try:
    from .s3_utils import (
        check_file_exists_in_s3,
        read_file_from_s3,
        upload_file_to_s3,
        s3_client,
    )

    S3_AVAILABLE = s3_client is not None
except ImportError:
    S3_AVAILABLE = False
    print("[WARNING] S3 utilities not available")

# Import directive analyzer for basic text extraction
try:
    from .directive_analyzer import (
        extract_full_text_from_html,
        is_xml_content,
        detect_language,
    )

    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("[WARNING] Directive analyzer not available")

# AWS Comprehend client
try:
    import boto3

    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")

    client_config = {"service_name": "comprehend", "region_name": aws_region}
    if aws_access_key and aws_secret_key:
        client_config["aws_access_key_id"] = aws_access_key
        client_config["aws_secret_access_key"] = aws_secret_key
    if aws_session_token:
        client_config["aws_session_token"] = aws_session_token

    comprehend_client = boto3.client(**client_config)

    # Bedrock Runtime client (for Claude/Titan inference)
    bedrock_config = {"service_name": "bedrock-runtime", "region_name": aws_region}
    if aws_access_key and aws_secret_key:
        bedrock_config["aws_access_key_id"] = aws_access_key
        bedrock_config["aws_secret_access_key"] = aws_secret_key
    if aws_session_token:
        bedrock_config["aws_session_token"] = aws_session_token

    bedrock_client = boto3.client(**bedrock_config)

    AWS_NLP_AVAILABLE = True
    print("[INFO] AWS Comprehend and Bedrock clients initialized")
except Exception as e:
    comprehend_client = None
    bedrock_client = None
    AWS_NLP_AVAILABLE = False
    print(f"[WARNING] AWS NLP services not available: {e}")

# spaCy fallback
try:
    import spacy

    try:
        nlp_spacy = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("[INFO] spaCy loaded successfully (fallback available)")
    except OSError:
        SPACY_AVAILABLE = False
        print(
            "[INFO] spaCy model not installed. Run: python -m spacy download en_core_web_sm"
        )
except ImportError:
    SPACY_AVAILABLE = False
    print("[INFO] spaCy not installed (will use AWS Comprehend only)")


class DirectiveNLPPipeline:
    """
    Lazy-processing two-layer NLP pipeline for regulatory directives.
    """

    def __init__(self, portfolio_name: str = "default"):
        """
        Initialize the NLP pipeline.

        Args:
            portfolio_name: Portfolio name for organizing extracted data
        """
        self.portfolio_name = portfolio_name
        self.extraction_prefix = f"data/extracted_directives/{portfolio_name}/"

    def check_existing_extraction(self, directive_filename: str) -> Optional[Dict]:
        """
        Check if extraction already exists in S3 and return it.

        Args:
            directive_filename: Original directive filename (e.g., "directive1.html")

        Returns:
            Existing extraction dict if found, None otherwise
        """
        if not S3_AVAILABLE:
            print("[WARNING] S3 not available, cannot check for existing extraction")
            return None

        # Clean filename for S3 key
        safe_name = re.sub(
            r"[^\w\s-]", "", directive_filename.replace(".html", "").replace(".xml", "")
        )[:50]

        # Look for any extraction file matching this directive
        # Pattern: {safe_name}_*_nlp_extraction.json
        try:
            from .s3_utils import list_files_in_s3

            existing_files = list_files_in_s3(self.extraction_prefix)

            # Find matching files
            matching_files = [
                f
                for f in existing_files
                if safe_name in f and f.endswith("_nlp_extraction.json")
            ]

            if matching_files:
                # Use the most recent one (last in sorted list)
                latest_file = sorted(matching_files)[-1]
                print(f"[INFO] Found existing NLP extraction: {latest_file}")

                # Download and parse
                content = read_file_from_s3(latest_file)
                extraction = json.loads(content)

                print(
                    f"[INFO] Reusing extraction from {extraction.get('metadata', {}).get('extraction_date', 'unknown date')}"
                )
                return extraction
            else:
                print(
                    f"[INFO] No existing NLP extraction found for: {directive_filename}"
                )
                return None

        except Exception as e:
            print(f"[WARNING] Error checking for existing extraction: {e}")
            return None

    def extract_entities_layer1_aws(self, text: str, language_code: str = "en") -> Dict:
        """
        Layer 1: Extract entities and key phrases using AWS Comprehend.

        Args:
            text: Directive text content
            language_code: Language code (en, es, fr, de, it, pt, etc.)

        Returns:
            Dictionary with entities and key phrases
        """
        if not AWS_NLP_AVAILABLE or not comprehend_client:
            raise RuntimeError("AWS Comprehend not available")

        # AWS Comprehend has a 5000 byte limit per request
        # Split text into chunks if needed
        max_bytes = 4500  # Leave some margin
        text_bytes = text.encode("utf-8")

        if len(text_bytes) > max_bytes:
            # Use first chunk for entity extraction
            text_chunk = text_bytes[:max_bytes].decode("utf-8", errors="ignore")
            print(
                f"[INFO] Text truncated to {len(text_chunk)} characters for AWS Comprehend"
            )
        else:
            text_chunk = text

        result = {
            "entities": [],
            "key_phrases": [],
            "sentiment": {},
            "syntax_tokens": [],
        }

        try:
            # Detect entities
            entities_response = comprehend_client.detect_entities(
                Text=text_chunk, LanguageCode=language_code
            )
            result["entities"] = entities_response.get("Entities", [])
            print(f"[INFO] AWS Comprehend detected {len(result['entities'])} entities")

            # Detect key phrases
            key_phrases_response = comprehend_client.detect_key_phrases(
                Text=text_chunk, LanguageCode=language_code
            )
            result["key_phrases"] = key_phrases_response.get("KeyPhrases", [])
            print(
                f"[INFO] AWS Comprehend detected {len(result['key_phrases'])} key phrases"
            )

            # Detect sentiment
            sentiment_response = comprehend_client.detect_sentiment(
                Text=text_chunk, LanguageCode=language_code
            )
            result["sentiment"] = {
                "sentiment": sentiment_response.get("Sentiment"),
                "scores": sentiment_response.get("SentimentScore", {}),
            }
            print(
                f"[INFO] AWS Comprehend sentiment: {result['sentiment']['sentiment']}"
            )

        except Exception as e:
            print(f"[ERROR] AWS Comprehend extraction failed: {e}")
            raise

        return result

    def extract_entities_layer1_spacy(self, text: str) -> Dict:
        """
        Layer 1: Extract entities and key phrases using spaCy (fallback).

        Args:
            text: Directive text content

        Returns:
            Dictionary with entities and key phrases
        """
        if not SPACY_AVAILABLE or not nlp_spacy:
            raise RuntimeError("spaCy not available")

        # Process with spaCy (limit to 1M characters)
        text_sample = text[:1000000] if len(text) > 1000000 else text
        doc = nlp_spacy(text_sample)

        result = {
            "entities": [],
            "key_phrases": [],
            "sentiment": {"sentiment": "NEUTRAL", "scores": {}},
            "syntax_tokens": [],
        }

        # Extract named entities
        for ent in doc.ents:
            result["entities"].append(
                {
                    "Text": ent.text,
                    "Type": ent.label_,
                    "Score": 0.9,  # spaCy doesn't provide confidence scores
                    "BeginOffset": ent.start_char,
                    "EndOffset": ent.end_char,
                }
            )

        # Extract noun phrases as key phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 5:  # Filter short phrases
                result["key_phrases"].append(
                    {
                        "Text": chunk.text,
                        "Score": 0.8,
                        "BeginOffset": chunk.start_char,
                        "EndOffset": chunk.end_char,
                    }
                )

        print(f"[INFO] spaCy detected {len(result['entities'])} entities")
        print(f"[INFO] spaCy detected {len(result['key_phrases'])} key phrases")

        return result

    def extract_entities_layer1(self, text: str, language_code: str = "en") -> Dict:
        """
        Layer 1: Extract entities and key phrases (auto-select AWS or spaCy).

        Args:
            text: Directive text content
            language_code: Language code

        Returns:
            Dictionary with entities and key phrases
        """
        # Try AWS Comprehend first
        if AWS_NLP_AVAILABLE:
            try:
                return self.extract_entities_layer1_aws(text, language_code)
            except Exception as e:
                print(f"[WARNING] AWS Comprehend failed, trying spaCy fallback: {e}")

        # Fallback to spaCy
        if SPACY_AVAILABLE:
            try:
                return self.extract_entities_layer1_spacy(text)
            except Exception as e:
                print(f"[ERROR] spaCy fallback also failed: {e}")
                raise

        raise RuntimeError(
            "No NLP service available (AWS Comprehend and spaCy both unavailable)"
        )

    def summarize_impact_layer2_bedrock(
        self, text: str, layer1_results: Dict, directive_metadata: Dict
    ) -> Dict:
        """
        Layer 2: Summarize regulatory impact using AWS Bedrock (Claude or Titan).

        Args:
            text: Directive text content
            layer1_results: Results from Layer 1 entity extraction
            directive_metadata: Metadata about the directive

        Returns:
            Dictionary with impact summary and analysis
        """
        if not AWS_NLP_AVAILABLE or not bedrock_client:
            raise RuntimeError("AWS Bedrock not available")

        # Prepare prompt for Bedrock
        entities_summary = ", ".join(
            [e["Text"] for e in layer1_results["entities"][:20]]
        )
        key_phrases_summary = ", ".join(
            [kp["Text"] for kp in layer1_results["key_phrases"][:20]]
        )

        prompt = f"""Analyze this regulatory directive and summarize its potential impact on financial markets and S&P 500 companies.

Directive Information:
Title: {directive_metadata.get('title', 'Unknown')}
Language: {directive_metadata.get('language', 'Unknown')}

Key Entities Detected: {entities_summary}

Key Phrases: {key_phrases_summary}

Text Sample (first 3000 chars):
{text[:3000]}

Provide a structured analysis covering:
1. Affected Industries/Sectors
2. Financial Impact Assessment (costs, penalties, compliance requirements)
3. Timeline and Implementation Deadlines
4. Risk Factors for Companies
5. Geographic Scope
6. Overall Market Impact Rating (Low/Medium/High)

Format as JSON with the following structure:
{{
  "affected_sectors": ["sector1", "sector2"],
  "financial_impact": {{
    "estimated_compliance_cost": "description",
    "penalties_range": "description",
    "revenue_impact": "description"
  }},
  "timeline": {{
    "effective_date": "date or description",
    "key_deadlines": ["deadline1", "deadline2"]
  }},
  "risk_factors": ["risk1", "risk2"],
  "geographic_scope": ["region1", "region2"],
  "market_impact_rating": "Low/Medium/High",
  "executive_summary": "2-3 sentence summary"
}}
"""

        try:
            # Use Claude 3 Haiku (cost-effective) or Claude 3 Sonnet (higher quality)
            model_id = "anthropic.claude-3-haiku-20240307-v1:0"

            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            }

            response = bedrock_client.invoke_model(
                modelId=model_id, body=json.dumps(request_body)
            )

            response_body = json.loads(response["body"].read())
            content = response_body["content"][0]["text"]

            print(f"[INFO] AWS Bedrock (Claude) generated impact summary")

            # Parse JSON from response
            try:
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    impact_summary = json.loads(json_match.group())
                else:
                    # If no JSON found, create structured response from text
                    impact_summary = {
                        "affected_sectors": [],
                        "financial_impact": {},
                        "timeline": {},
                        "risk_factors": [],
                        "geographic_scope": [],
                        "market_impact_rating": "Medium",
                        "executive_summary": content[:500],
                    }
            except json.JSONDecodeError:
                impact_summary = {
                    "affected_sectors": [],
                    "financial_impact": {},
                    "timeline": {},
                    "risk_factors": [],
                    "geographic_scope": [],
                    "market_impact_rating": "Medium",
                    "executive_summary": content[:500],
                    "raw_response": content,
                }

            return impact_summary

        except Exception as e:
            print(f"[ERROR] AWS Bedrock Claude invocation failed: {e}")
            # Try fallback to Titan
            try:
                return self._summarize_with_titan_fallback(
                    text, layer1_results, directive_metadata
                )
            except Exception as titan_error:
                print(f"[ERROR] Titan fallback also failed: {titan_error}")
                raise

    def _summarize_with_titan_fallback(
        self, text: str, layer1_results: Dict, directive_metadata: Dict
    ) -> Dict:
        """
        Fallback to AWS Titan for impact summarization.
        """
        entities_summary = ", ".join(
            [e["Text"] for e in layer1_results["entities"][:20]]
        )

        prompt = f"""Analyze regulatory directive impact on financial markets.

Title: {directive_metadata.get('title', 'Unknown')}
Key Entities: {entities_summary}

Text: {text[:2000]}

Summarize: affected sectors, financial impact, timeline, risks, geographic scope, market impact rating.
"""

        model_id = "amazon.titan-text-express-v1"

        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1500,
                "temperature": 0.3,
                "topP": 0.9,
            },
        }

        response = bedrock_client.invoke_model(
            modelId=model_id, body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        content = response_body["results"][0]["outputText"]

        print(f"[INFO] AWS Bedrock (Titan) generated impact summary")

        return {
            "affected_sectors": [],
            "financial_impact": {},
            "timeline": {},
            "risk_factors": [],
            "geographic_scope": [],
            "market_impact_rating": "Medium",
            "executive_summary": content[:500],
            "raw_response": content,
        }

    def summarize_impact_layer2_openai(
        self, text: str, layer1_results: Dict, directive_metadata: Dict
    ) -> Dict:
        """
        Layer 2: Summarize regulatory impact using OpenAI (fallback).

        Args:
            text: Directive text content
            layer1_results: Results from Layer 1 entity extraction
            directive_metadata: Metadata about the directive

        Returns:
            Dictionary with impact summary
        """
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise RuntimeError("OpenAI API key not configured")

            client = openai.OpenAI(api_key=api_key)

            entities_summary = ", ".join(
                [e["Text"] for e in layer1_results["entities"][:20]]
            )
            key_phrases_summary = ", ".join(
                [kp["Text"] for kp in layer1_results["key_phrases"][:20]]
            )

            prompt = f"""Analyze this regulatory directive and provide a structured impact assessment.

Directive: {directive_metadata.get('title', 'Unknown')}
Key Entities: {entities_summary}
Key Phrases: {key_phrases_summary}

Text Sample: {text[:3000]}

Provide JSON output with: affected_sectors, financial_impact, timeline, risk_factors, geographic_scope, market_impact_rating, executive_summary.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial regulatory analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.3,
            )

            content = response.choices[0].message.content

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "executive_summary": content[:500],
                    "market_impact_rating": "Medium",
                }

        except Exception as e:
            print(f"[ERROR] OpenAI summarization failed: {e}")
            raise

    def summarize_impact_layer2(
        self, text: str, layer1_results: Dict, directive_metadata: Dict
    ) -> Dict:
        """
        Layer 2: Summarize regulatory impact (auto-select AWS Bedrock or OpenAI).

        Args:
            text: Directive text content
            layer1_results: Results from Layer 1
            directive_metadata: Metadata about the directive

        Returns:
            Dictionary with impact summary
        """
        # Try AWS Bedrock first
        if AWS_NLP_AVAILABLE and bedrock_client:
            try:
                return self.summarize_impact_layer2_bedrock(
                    text, layer1_results, directive_metadata
                )
            except Exception as e:
                print(f"[WARNING] AWS Bedrock failed, trying OpenAI fallback: {e}")

        # Fallback to OpenAI
        try:
            return self.summarize_impact_layer2_openai(
                text, layer1_results, directive_metadata
            )
        except Exception as e:
            print(f"[ERROR] OpenAI fallback also failed: {e}")
            # Return minimal structure
            return {
                "affected_sectors": [],
                "financial_impact": {},
                "timeline": {},
                "risk_factors": [],
                "geographic_scope": [],
                "market_impact_rating": "Unknown",
                "executive_summary": "Impact analysis unavailable - no NLP service configured",
                "error": str(e),
            }

    def process_directive(
        self,
        directive_path: str,
        directive_content: str = None,
        force_reprocess: bool = False,
    ) -> Dict:
        """
        Process a directive through the complete two-layer NLP pipeline.
        Lazy loads existing extractions to avoid redundant processing.

        Args:
            directive_path: Path to directive file (for identification)
            directive_content: Optional pre-loaded directive content (HTML/XML)
            force_reprocess: If True, reprocess even if extraction exists

        Returns:
            Complete extraction results
        """
        directive_filename = os.path.basename(directive_path)

        # Step 1: Check for existing extraction (lazy loading)
        if not force_reprocess:
            existing = self.check_existing_extraction(directive_filename)
            if existing:
                return existing

        print(f"[INFO] Processing directive with NLP pipeline: {directive_filename}")

        # Step 2: Load and extract text
        if not directive_content:
            if not os.path.exists(directive_path):
                raise FileNotFoundError(f"Directive file not found: {directive_path}")
            with open(directive_path, "r", encoding="utf-8", errors="ignore") as f:
                directive_content = f.read()

        # Detect format and extract text
        is_xml = is_xml_content(directive_content) if ANALYZER_AVAILABLE else False
        text = (
            extract_full_text_from_html(directive_content, is_xml=is_xml)
            if ANALYZER_AVAILABLE
            else directive_content
        )

        if not text or len(text) < 100:
            raise ValueError("Insufficient text extracted from directive")

        # Detect language
        if ANALYZER_AVAILABLE:
            language_code, confidence = detect_language(text)
        else:
            language_code, confidence = "en", 0.8

        print(f"[INFO] Language: {language_code} (confidence: {confidence:.2f})")
        print(f"[INFO] Text length: {len(text)} characters")

        # Step 3: Layer 1 - Entity and key phrase extraction
        print("[INFO] Running Layer 1: Entity extraction...")
        layer1_results = self.extract_entities_layer1(text, language_code)

        # Prepare metadata
        directive_metadata = {
            "filename": directive_filename,
            "title": text[:200].split("\n")[0].strip(),
            "language": language_code,
            "language_confidence": confidence,
            "text_length": len(text),
        }

        # Step 4: Layer 2 - Impact summarization
        print("[INFO] Running Layer 2: Impact summarization...")
        layer2_results = self.summarize_impact_layer2(
            text, layer1_results, directive_metadata
        )

        # Step 5: Combine results
        complete_extraction = {
            "metadata": {
                "directive_filename": directive_filename,
                "directive_path": directive_path,
                "extraction_date": datetime.now().isoformat(),
                "language": language_code,
                "language_confidence": confidence,
                "text_length": len(text),
                "pipeline_version": "2.0",
                "nlp_services_used": {
                    "layer1": "AWS Comprehend" if AWS_NLP_AVAILABLE else "spaCy",
                    "layer2": (
                        "AWS Bedrock Claude"
                        if (AWS_NLP_AVAILABLE and bedrock_client)
                        else "OpenAI"
                    ),
                },
            },
            "layer1_extraction": {
                "entities": layer1_results["entities"][:50],  # Limit to top 50
                "key_phrases": layer1_results["key_phrases"][:50],
                "sentiment": layer1_results.get("sentiment", {}),
                "total_entities": len(layer1_results["entities"]),
                "total_key_phrases": len(layer1_results["key_phrases"]),
            },
            "layer2_impact_summary": layer2_results,
            "text_sample": text[:2000],  # Include sample for reference
        }

        # Step 6: Save to S3
        if S3_AVAILABLE:
            try:
                self._save_extraction_to_s3(directive_filename, complete_extraction)
            except Exception as e:
                print(f"[WARNING] Could not save extraction to S3: {e}")

        print(f"[SUCCESS] NLP pipeline completed for {directive_filename}")
        return complete_extraction

    def _save_extraction_to_s3(self, directive_filename: str, extraction: Dict) -> str:
        """
        Save NLP extraction results to S3.

        Args:
            directive_filename: Original directive filename
            extraction: Complete extraction results

        Returns:
            S3 key of saved file
        """
        # Clean filename
        safe_name = re.sub(
            r"[^\w\s-]", "", directive_filename.replace(".html", "").replace(".xml", "")
        )[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{safe_name}_{timestamp}_nlp_extraction.json"
        s3_key = f"{self.extraction_prefix}{filename}"

        # Convert to JSON
        json_content = json.dumps(extraction, indent=2, ensure_ascii=False)
        json_bytes = json_content.encode("utf-8")

        # Upload
        upload_file_to_s3(json_bytes, s3_key, overwrite=False)

        file_size_kb = len(json_bytes) / 1024
        print(f"[INFO] Saved NLP extraction to S3: {s3_key} ({file_size_kb:.2f} KB)")

        return s3_key

    def batch_process_directives(
        self, directive_paths: List[str], force_reprocess: bool = False
    ) -> Dict[str, Dict]:
        """
        Process multiple directives with lazy loading.

        Args:
            directive_paths: List of directive file paths
            force_reprocess: If True, reprocess all directives

        Returns:
            Dictionary mapping filename -> extraction results
        """
        results = {}

        for directive_path in directive_paths:
            filename = os.path.basename(directive_path)
            try:
                extraction = self.process_directive(
                    directive_path, force_reprocess=force_reprocess
                )
                results[filename] = extraction
                print(f"[SUCCESS] Processed {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
                results[filename] = {
                    "error": str(e),
                    "metadata": {"directive_filename": filename},
                }

        print(f"[INFO] Batch processing complete: {len(results)} directives processed")
        return results


# Convenience functions
def process_directive_with_nlp(
    directive_path: str, portfolio_name: str = "default", force_reprocess: bool = False
) -> Dict:
    """
    Process a single directive through the NLP pipeline.

    Args:
        directive_path: Path to directive file
        portfolio_name: Portfolio name for organization
        force_reprocess: If True, reprocess even if extraction exists

    Returns:
        Complete extraction results
    """
    pipeline = DirectiveNLPPipeline(portfolio_name=portfolio_name)
    return pipeline.process_directive(directive_path, force_reprocess=force_reprocess)


def batch_process_directives_with_nlp(
    directive_directory: str,
    portfolio_name: str = "default",
    force_reprocess: bool = False,
) -> Dict[str, Dict]:
    """
    Process all directives in a directory through the NLP pipeline.

    Args:
        directive_directory: Directory containing directive files
        portfolio_name: Portfolio name for organization
        force_reprocess: If True, reprocess all directives

    Returns:
        Dictionary mapping filename -> extraction results
    """
    import glob

    html_files = glob.glob(os.path.join(directive_directory, "*.html"))
    xml_files = glob.glob(os.path.join(directive_directory, "*.xml"))
    all_files = html_files + xml_files

    print(f"[INFO] Found {len(all_files)} directive files to process")

    pipeline = DirectiveNLPPipeline(portfolio_name=portfolio_name)
    return pipeline.batch_process_directives(all_files, force_reprocess=force_reprocess)
