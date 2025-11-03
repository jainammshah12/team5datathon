"""
Lazy-Processing Two-Layer NLP Pipeline for Regulatory Directives

This module implements an intelligent NLP pipeline that:
1. Checks if extraction already exists in S3 before processing
2. Layer 1: spaCy for entity/key phrase extraction
3. Layer 2: OpenAI/Perplexity for impact summarization

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

# AWS services removed - using only spaCy and OpenAI/Perplexity

# spaCy for entity extraction
try:
    import spacy

    try:
        nlp_spacy = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("[INFO] spaCy loaded successfully for entity extraction")
    except OSError:
        SPACY_AVAILABLE = False
        print(
            "[INFO] spaCy model not installed. Run: python -m spacy download en_core_web_sm"
        )
except ImportError:
    SPACY_AVAILABLE = False
    print("[INFO] spaCy not installed - entity extraction will be limited")


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


    def extract_entities_layer1_spacy(self, text: str) -> Dict:
        """
        Layer 1: Extract entities and key phrases using spaCy.

        Args:
            text: Directive text content

        Returns:
            Dictionary with entities and key phrases
        """
        if not SPACY_AVAILABLE or not nlp_spacy:
            raise RuntimeError("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

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
        Layer 1: Extract entities and key phrases using spaCy.

        Args:
            text: Directive text content
            language_code: Language code (not used with spaCy)

        Returns:
            Dictionary with entities and key phrases
        """
        if SPACY_AVAILABLE:
            return self.extract_entities_layer1_spacy(text)
        else:
            raise RuntimeError(
                "spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )


    def summarize_impact_layer2_openai(
        self, text: str, layer1_results: Dict, directive_metadata: Dict
    ) -> Dict:
        """
        Layer 2: Summarize regulatory impact using OpenAI/Perplexity.

        Args:
            text: Directive text content
            layer1_results: Results from Layer 1 entity extraction
            directive_metadata: Metadata about the directive

        Returns:
            Dictionary with impact summary
        """
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PERPLEXITY_API_KEY")

            if not api_key:
                raise RuntimeError("OpenAI or Perplexity API key not configured")

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
        Layer 2: Summarize regulatory impact using OpenAI/Perplexity.

        Args:
            text: Directive text content
            layer1_results: Results from Layer 1
            directive_metadata: Metadata about the directive

        Returns:
            Dictionary with impact summary
        """
        try:
            return self.summarize_impact_layer2_openai(
                text, layer1_results, directive_metadata
            )
        except Exception as e:
            print(f"[ERROR] Impact summarization failed: {e}")
            # Return minimal structure
            return {
                "affected_sectors": [],
                "financial_impact": {},
                "timeline": {},
                "risk_factors": [],
                "geographic_scope": [],
                "market_impact_rating": "Unknown",
                "executive_summary": "Impact analysis unavailable - OpenAI/Perplexity API key required",
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
                    "layer1": "spaCy",
                    "layer2": "OpenAI/Perplexity",
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
