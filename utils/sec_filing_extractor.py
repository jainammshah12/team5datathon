"""
SEC Filing Extractor using sec-parser and BeautifulSoup
Parses 10-K and 10-Q HTML/XML files and extracts key sections with full text content
"""

import os
import json
import glob
from typing import Dict, List, Optional
from datetime import datetime

try:
    from sec_parser import Filing

    SEC_PARSER_AVAILABLE = True
except ImportError:
    SEC_PARSER_AVAILABLE = False
    print("[WARNING] sec-parser not available. Install with: pip install sec-parser")
from bs4 import BeautifulSoup

# Import S3 utilities
try:
    from .s3_utils import (
        upload_file_to_s3,
        list_files_in_s3,
        delete_file_from_s3,
        check_file_exists_in_s3,
    )

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("[WARNING] S3 utilities not available. Will save locally only.")


def is_xml_content(content: str) -> bool:
    """
    Detect if content is XML format.

    Args:
        content: File content string

    Returns:
        True if content appears to be XML
    """
    content_stripped = content.strip()
    # Check for XML declaration or common XML patterns
    return (
        content_stripped.startswith("<?xml")
        or content_stripped.startswith("<XML>")
        or "<XML>" in content_stripped[:500]
    )


def extract_full_text_from_html(html_content: str, is_xml: bool = False) -> str:
    """
    Extract full text content from HTML or XML using BeautifulSoup.

    Args:
        html_content: HTML/XML content of the filing
        is_xml: If True, use XML parser to maintain case sensitivity

    Returns:
        Full text content
    """
    try:
        # Choose appropriate parser - XML parser for XML files, lxml for HTML
        parser = "xml" if is_xml else "lxml"
        soup = BeautifulSoup(html_content, parser)

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text - preserve structure with separator
        text = soup.get_text(separator="\n", strip=True)

        # Clean up multiple newlines while preserving paragraph structure
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return clean_text
    except Exception as e:
        print(f"[ERROR] Failed to extract text with BeautifulSoup: {e}")
        return ""


def parse_sec_filing(html_content: str, filing_type: str = "10-K") -> tuple:
    """
    Parse SEC filing HTML content using sec-parser.

    Args:
        html_content: HTML content of the filing
        filing_type: Type of filing (10-K or 10-Q)

    Returns:
        Tuple of (filing object, list of semantic elements)
    """
    try:
        if not SEC_PARSER_AVAILABLE:
            print(f"[WARNING] sec-parser not available, skipping semantic parsing")
            return None, []

        # Create Filing object
        filing = Filing(html_content)

        # Parse the filing to get semantic elements
        elements = filing.parse()

        print(
            f"[INFO] Successfully parsed {filing_type} filing with {len(elements) if elements else 0} semantic elements"
        )
        return filing, elements if elements else []
    except Exception as e:
        print(f"[ERROR] Failed to parse filing with sec-parser: {e}")
        return None, []


def extract_element_text(element) -> str:
    """
    Extract full text from a sec-parser element.

    Args:
        element: Parsed semantic element from sec-parser

    Returns:
        Full text content of the element
    """
    try:
        # Try different methods to extract text from the element
        # sec-parser elements typically have a 'text' property
        if hasattr(element, "text"):
            text = element.text
            if text and isinstance(text, str):
                return text.strip()

        # Try get_text() method if available
        if hasattr(element, "get_text"):
            text = element.get_text()
            if text and isinstance(text, str):
                return text.strip()

        # Try to access inner_html or html content
        if hasattr(element, "inner_html"):
            soup = BeautifulSoup(str(element.inner_html), "html.parser")
            return soup.get_text(separator=" ", strip=True)

        # Try html_tag attribute
        if hasattr(element, "html_tag"):
            soup = BeautifulSoup(str(element.html_tag), "html.parser")
            return soup.get_text(separator=" ", strip=True)

        # Fallback to string representation
        return str(element).strip()
    except Exception as e:
        print(f"[WARNING] Failed to extract text from element: {e}")
        return ""


def render_parsed_elements(elements: list) -> str:
    """
    Render parsed elements to readable text with full content.

    Args:
        elements: List of parsed semantic elements from sec-parser

    Returns:
        Rendered text output with full content - no information loss
    """
    try:
        output_lines = []

        for i, element in enumerate(elements):
            # Get element type
            element_type = type(element).__name__

            # Extract full text from the element
            text = extract_element_text(element)

            if text:
                # Include element metadata if available
                metadata = []
                if hasattr(element, "identifier"):
                    metadata.append(f"ID: {element.identifier}")
                if hasattr(element, "level"):
                    metadata.append(f"Level: {element.level}")

                metadata_str = f" ({', '.join(metadata)})" if metadata else ""
                output_lines.append(f"[Element {i+1} - {element_type}{metadata_str}]")
                output_lines.append(text)
                output_lines.append("")  # Add blank line for readability

        return "\n".join(output_lines)
    except Exception as e:
        print(f"[ERROR] Failed to render elements: {e}")
        return ""


def extract_sections_from_html(
    html_content: str, is_xml: bool = False
) -> Dict[str, str]:
    """
    Extract specific sections from HTML/XML using BeautifulSoup and pattern matching.
    Ensures no information is lost during extraction.

    Args:
        html_content: HTML/XML content of the filing
        is_xml: If True, parse as XML to preserve case sensitivity

    Returns:
        Dictionary with extracted sections
    """
    sections = {
        # Core Business & Risk Sections (High impact on stock value)
        "business": "",  # Item 1: Business operations, strategy
        "risk_factors": "",  # Item 1A: Regulatory & business risks
        "cybersecurity": "",  # Item 1C: Cyber incidents & governance (NEW 2023)
        "properties": "",  # Item 2: Facility locations, jurisdictions
        "legal_proceedings": "",  # Item 3: Litigation, regulatory actions
        # Financial Analysis Sections (Direct stock impact)
        "market_for_equity": "",  # Item 5: Buybacks, dividends, shareholder returns
        "mda": "",  # Item 7: Financial discussion, outlook
        "market_risk": "",  # Item 7A: Financial risk exposures
        "financial_statements": "",  # Item 8: Financial data, notes
        # Governance & Control Sections (Risk indicators)
        "controls_and_procedures": "",  # Item 9A: Internal controls, SOX compliance
        "executive_compensation": "",  # Item 11: Exec pay, incentive alignment
        # Material Events (8-K sections - immediate stock impact)
        "material_agreements": "",  # 8-K 1.01: Major contracts, partnerships
        "cybersecurity_incidents": "",  # 8-K 1.05: Material cyber incidents
        "acquisitions_dispositions": "",  # 8-K 2.01: M&A activity
        "earnings_announcements": "",  # 8-K 2.02: Financial results
        "material_impairments": "",  # 8-K 2.06: Asset write-downs
        "accountant_changes": "",  # 8-K 4.01: Auditor changes (red flag)
        "control_changes": "",  # 8-K 5.01: Ownership changes
        "officer_director_changes": "",  # 8-K 5.02: Leadership changes
        "full_text": "",
    }

    try:
        # Detect if content is XML
        if not is_xml and is_xml_content(html_content):
            is_xml = True
            print("[INFO] Detected XML format, using XML parser")

        # Extract full text using BeautifulSoup with appropriate parser
        sections["full_text"] = extract_full_text_from_html(html_content, is_xml=is_xml)

        if not sections["full_text"]:
            print(f"[WARNING] No text extracted from filing")
            return sections

        print(f"[INFO] Extracted full text: {len(sections['full_text'])} characters")

        # Split into lines for section detection
        lines = sections["full_text"].split("\n")

        current_section = None
        section_lines = []

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Detect section headers with improved pattern matching
            # Item 1: Business (but not Item 1A, 1B)
            if (
                (
                    "item 1" in line_lower
                    or "item 1." in line_lower
                    or "item1" in line_lower
                )
                and "business" in line_lower
                and "item 1a" not in line_lower
                and "item 1.a" not in line_lower
                and "item1a" not in line_lower
                and "item 1b" not in line_lower
                and "item 1.b" not in line_lower
                and "item1b" not in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "business"
                section_lines = [line]
                print(f"[INFO] Found Business section at line {i}")

            # Item 1A: Risk Factors
            elif (
                "item 1a" in line_lower
                or "item 1.a" in line_lower
                or "item1a" in line_lower
            ) and "risk" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "risk_factors"
                section_lines = [line]
                print(f"[INFO] Found Risk Factors section at line {i}")

            # Item 1C: Cybersecurity (NEW 2023)
            elif (
                "item 1c" in line_lower
                or "item 1.c" in line_lower
                or "item1c" in line_lower
            ) and "cyber" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "cybersecurity"
                section_lines = [line]
                print(f"[INFO] Found Cybersecurity section at line {i}")

            # Item 2: Properties
            elif (
                "item 2" in line_lower
                or "item 2." in line_lower
                or "item2" in line_lower
            ) and "propert" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "properties"
                section_lines = [line]
                print(f"[INFO] Found Properties section at line {i}")

            # Item 3: Legal Proceedings
            elif (
                "item 3" in line_lower
                or "item 3." in line_lower
                or "item3" in line_lower
            ) and "legal" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "legal_proceedings"
                section_lines = [line]
                print(f"[INFO] Found Legal Proceedings section at line {i}")

            # Item 5: Market for Equity (Stock buybacks, dividends)
            elif (
                "item 5" in line_lower
                or "item 5." in line_lower
                or "item5" in line_lower
            ) and (
                "market" in line_lower
                or "equity" in line_lower
                or "stockholder" in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "market_for_equity"
                section_lines = [line]
                print(f"[INFO] Found Market for Equity section at line {i}")

            # Item 7: MD&A (Management's Discussion and Analysis)
            elif (
                (
                    "item 7" in line_lower
                    or "item 7." in line_lower
                    or "item7" in line_lower
                )
                and (
                    "management" in line_lower
                    or "discussion" in line_lower
                    or "mda" in line_lower
                    or "md&a" in line_lower
                )
                and "item 7a" not in line_lower
                and "item 7.a" not in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "mda"
                section_lines = [line]
                print(f"[INFO] Found MD&A section at line {i}")

            # Item 7A: Market Risk
            elif (
                "item 7a" in line_lower
                or "item 7.a" in line_lower
                or "item7a" in line_lower
            ) and ("market" in line_lower or "quantitative" in line_lower):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "market_risk"
                section_lines = [line]
                print(f"[INFO] Found Market Risk section at line {i}")

            # Item 8: Financial Statements
            elif (
                "item 8" in line_lower
                or "item 8." in line_lower
                or "item8" in line_lower
            ) and ("financial" in line_lower or "statements" in line_lower):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "financial_statements"
                section_lines = [line]
                print(f"[INFO] Found Financial Statements section at line {i}")

            # Item 9A: Controls and Procedures
            elif (
                "item 9a" in line_lower
                or "item 9.a" in line_lower
                or "item9a" in line_lower
            ) and ("control" in line_lower or "procedure" in line_lower):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "controls_and_procedures"
                section_lines = [line]
                print(f"[INFO] Found Controls and Procedures section at line {i}")

            # Item 11: Executive Compensation
            elif (
                "item 11" in line_lower
                or "item 11." in line_lower
                or "item11" in line_lower
            ) and ("executive" in line_lower or "compensation" in line_lower):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "executive_compensation"
                section_lines = [line]
                print(f"[INFO] Found Executive Compensation section at line {i}")

            # 8-K Material Events Detection
            # 8-K 1.01: Material Agreements
            elif ("1.01" in line_lower or "item 1.01" in line_lower) and (
                "agreement" in line_lower or "definitive" in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "material_agreements"
                section_lines = [line]
                print(f"[INFO] Found Material Agreements (8-K 1.01) at line {i}")

            # 8-K 1.05: Cybersecurity Incidents
            elif (
                "1.05" in line_lower or "item 1.05" in line_lower
            ) and "cyber" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "cybersecurity_incidents"
                section_lines = [line]
                print(f"[INFO] Found Cybersecurity Incident (8-K 1.05) at line {i}")

            # 8-K 2.01: Acquisitions/Dispositions
            elif ("2.01" in line_lower or "item 2.01" in line_lower) and (
                "acquisition" in line_lower or "disposition" in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "acquisitions_dispositions"
                section_lines = [line]
                print(f"[INFO] Found Acquisition/Disposition (8-K 2.01) at line {i}")

            # 8-K 2.02: Financial Results/Earnings
            elif ("2.02" in line_lower or "item 2.02" in line_lower) and (
                "results" in line_lower or "financial" in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "earnings_announcements"
                section_lines = [line]
                print(f"[INFO] Found Earnings Announcement (8-K 2.02) at line {i}")

            # 8-K 2.06: Material Impairments
            elif (
                "2.06" in line_lower or "item 2.06" in line_lower
            ) and "impairment" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "material_impairments"
                section_lines = [line]
                print(f"[INFO] Found Material Impairments (8-K 2.06) at line {i}")

            # 8-K 4.01: Accountant Changes
            elif (
                "4.01" in line_lower or "item 4.01" in line_lower
            ) and "accountant" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "accountant_changes"
                section_lines = [line]
                print(f"[INFO] Found Accountant Changes (8-K 4.01) at line {i}")

            # 8-K 5.01: Control Changes
            elif (
                "5.01" in line_lower or "item 5.01" in line_lower
            ) and "control" in line_lower:
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "control_changes"
                section_lines = [line]
                print(f"[INFO] Found Control Changes (8-K 5.01) at line {i}")

            # 8-K 5.02: Officer/Director Changes
            elif ("5.02" in line_lower or "item 5.02" in line_lower) and (
                "officer" in line_lower or "director" in line_lower
            ):
                if current_section and section_lines:
                    sections[current_section] = "\n".join(section_lines)
                current_section = "officer_director_changes"
                section_lines = [line]
                print(f"[INFO] Found Officer/Director Changes (8-K 5.02) at line {i}")

            # Stop at next Item to avoid overlapping sections (more lenient threshold)
            elif (
                current_section
                and line_lower.startswith("item ")
                and len(section_lines) > 20
            ):
                # Check if it's a new item section we're not tracking
                is_new_tracked_item = any(
                    [
                        "item 1" in line_lower and "business" in line_lower,
                        "item 1a" in line_lower and "risk" in line_lower,
                        "item 1c" in line_lower and "cyber" in line_lower,
                        "item 2" in line_lower and "propert" in line_lower,
                        "item 3" in line_lower and "legal" in line_lower,
                        "item 5" in line_lower
                        and ("market" in line_lower or "equity" in line_lower),
                        "item 7" in line_lower
                        and ("management" in line_lower or "discussion" in line_lower),
                        "item 7a" in line_lower and "market" in line_lower,
                        "item 8" in line_lower and "financial" in line_lower,
                        "item 9a" in line_lower and "control" in line_lower,
                        "item 11" in line_lower
                        and ("executive" in line_lower or "compensation" in line_lower),
                    ]
                )
                if not is_new_tracked_item:
                    # Save current section and stop tracking
                    sections[current_section] = "\n".join(section_lines)
                    current_section = None
                    section_lines = []

            elif current_section:
                section_lines.append(line)

        # Save last section if any
        if current_section and section_lines:
            sections[current_section] = "\n".join(section_lines)

        # Print statistics for extracted sections
        for section_name, content in sections.items():
            if section_name != "full_text" and content:
                word_count = len(content.split())
                line_count = len(content.split("\n"))
                char_count = len(content)
                print(
                    f"[INFO] Extracted {section_name}: {char_count} chars, {word_count} words, {line_count} lines"
                )

    except Exception as e:
        print(f"[ERROR] Failed to extract sections: {e}")
        import traceback

        traceback.print_exc()

    return sections


def save_extracted_sections_to_s3(
    ticker: str,
    sections: Dict[str, str],
    portfolio_name: str,
    filing_type: str = "10-K",
) -> str:
    """
    Save extracted SEC filing sections to S3 in a single comprehensive JSON file.
    Includes all key sections for regulatory analysis (full_text excluded to keep file size manageable).
    Automatically removes old extraction files to keep only the latest version.

    Note: full_text is NOT saved in the JSON file to reduce file size, but statistics
    about full_text length are included.

    Args:
        ticker: Company ticker symbol
        sections: Dictionary with extracted sections (including full_text for stats)
        portfolio_name: Portfolio name to organize files in S3
        filing_type: Type of filing (10-K, 10-Q, etc.)

    Returns:
        S3 key (path) of saved JSON file
    """
    if not S3_AVAILABLE:
        raise ConnectionError("S3 utilities not available. Cannot save to S3.")

    # Clean up old extraction files for this ticker and filing type in S3
    # This prevents duplicate files and ensures only the latest extraction is kept
    filing_type_normalized = filing_type.replace("-", "_")
    s3_prefix = f"extracted_filings/{ticker}/"

    try:
        # List existing files in S3 for this ticker
        existing_files = list_files_in_s3(s3_prefix)
        # Filter for files matching this filing type
        matching_files = [
            f
            for f in existing_files
            if filing_type_normalized in f and f.endswith(".json")
        ]

        if matching_files:
            print(
                f"[INFO] Cleaning up {len(matching_files)} old extraction file(s) from S3..."
            )
            for old_file in matching_files:
                try:
                    delete_file_from_s3(old_file)
                    print(f"[INFO] Removed from S3: {old_file.split('/')[-1]}")
                except Exception as e:
                    print(f"[WARNING] Could not remove old S3 file {old_file}: {e}")
    except Exception as e:
        print(f"[WARNING] Could not clean up old S3 files: {e}")

    # Prepare comprehensive output data with all sections (excluding full_text to keep file size manageable)
    output_data = {
        "metadata": {
            "ticker": ticker,
            "filing_type": filing_type,
            "extraction_date": datetime.now().isoformat(),
            "extractor_version": "2.0",
        },
        "sections": {
            # Core business & risk sections
            "business": sections.get("business", ""),
            "risk_factors": sections.get("risk_factors", ""),
            "cybersecurity": sections.get("cybersecurity", ""),
            "properties": sections.get("properties", ""),
            "legal_proceedings": sections.get("legal_proceedings", ""),
            # Financial analysis sections
            "market_for_equity": sections.get("market_for_equity", ""),
            "mda": sections.get("mda", ""),
            "market_risk": sections.get("market_risk", ""),
            "financial_statements": sections.get("financial_statements", ""),
            # Governance & control sections
            "controls_and_procedures": sections.get("controls_and_procedures", ""),
            "executive_compensation": sections.get("executive_compensation", ""),
            # Material events (8-K sections)
            "material_agreements": sections.get("material_agreements", ""),
            "cybersecurity_incidents": sections.get("cybersecurity_incidents", ""),
            "acquisitions_dispositions": sections.get("acquisitions_dispositions", ""),
            "earnings_announcements": sections.get("earnings_announcements", ""),
            "material_impairments": sections.get("material_impairments", ""),
            "accountant_changes": sections.get("accountant_changes", ""),
            "control_changes": sections.get("control_changes", ""),
            "officer_director_changes": sections.get("officer_director_changes", ""),
        },
        "statistics": {
            "full_text_length": len(sections.get("full_text", "")),
            "sections_found": [
                k
                for k in sections.keys()
                if sections.get(k) and k != "full_text" and not k.startswith("_")
            ],
            "total_sections_count": len(
                [
                    k
                    for k in sections.keys()
                    if sections.get(k) and k != "full_text" and not k.startswith("_")
                ]
            ),
        },
        "portfolio_impact_focus": {
            "description": "Sections optimized for portfolio recommendations and stock value impact analysis",
            "high_impact_sections": [
                "earnings_announcements - immediate stock price impact",
                "acquisitions_dispositions - M&A activity affects valuations",
                "material_impairments - asset write-downs signal trouble",
                "accountant_changes - red flag for accounting issues",
                "officer_director_changes - leadership stability",
                "cybersecurity_incidents - operational and reputation risk",
            ],
            "risk_indicators": [
                "risk_factors - regulatory and business risks",
                "legal_proceedings - litigation costs and regulatory actions",
                "controls_and_procedures - governance quality",
                "cybersecurity - cyber risk management",
            ],
            "value_drivers": [
                "market_for_equity - buybacks and dividends",
                "executive_compensation - management incentives",
                "mda - financial outlook and trends",
                "financial_statements - actual financial performance",
            ],
        },
    }

    # Upload to S3 as single comprehensive JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filing_type.replace('-', '_')}_{timestamp}_complete.json"
    s3_key = f"extracted_filings/{ticker}/{filename}"

    # Convert to JSON string and encode to bytes
    json_content = json.dumps(output_data, indent=2, ensure_ascii=False)
    json_bytes = json_content.encode("utf-8")

    # Upload to S3
    try:
        upload_file_to_s3(json_bytes, s3_key, overwrite=False)
        file_size_kb = len(json_bytes) / 1024
        print(f"[INFO] Saved complete filing to S3: {s3_key}")
        print(f"[INFO] File size: {file_size_kb:.2f} KB")
        print(
            f"[INFO] Sections extracted: {', '.join(output_data['statistics']['sections_found'])}"
        )
        return s3_key
    except Exception as e:
        print(f"[ERROR] Failed to upload to S3: {e}")
        raise


def extract_filing_from_html(
    ticker: str,
    html_content: str,
    portfolio_name: str,
    filing_type: str = "10-K",
    save_to_s3: bool = True,
    is_xml: bool = False,
) -> Dict[str, str]:
    """
    Extract key sections from SEC filing HTML/XML content and save to S3.
    Ensures no information is lost during extraction.

    Args:
        ticker: Company ticker symbol
        html_content: HTML/XML content of the SEC filing
        portfolio_name: Portfolio name to organize files in S3
        filing_type: Type of filing (10-K, 10-Q, etc.)
        save_to_s3: If True, save extracted data to S3 (default: True)
        is_xml: If True, parse as XML instead of HTML

    Returns:
        Dictionary with extracted sections and metadata
    """
    try:
        print(
            f"[INFO] Processing {filing_type} filing for {ticker} (Portfolio: {portfolio_name})..."
        )
        print(f"[INFO] Content size: {len(html_content)} characters")

        # Extract sections using BeautifulSoup
        sections = extract_sections_from_html(html_content, is_xml=is_xml)

        if not sections.get("full_text"):
            print(f"[WARNING] No text extracted from filing")
            return {"_extraction_success": False, "_error": "No text extracted"}

        # Try to use sec-parser for enhanced parsing (optional, provides semantic structure)
        try:
            filing_obj, elements = parse_sec_filing(html_content, filing_type)
            sections["_parsed_elements_count"] = len(elements)

            # Optionally save rendered elements for debugging
            if elements:
                rendered = render_parsed_elements(elements)
                if rendered:
                    sections["_parsed_content"] = rendered
                    print(
                        f"[INFO] Generated rendered content from {len(elements)} semantic elements"
                    )
        except Exception as e:
            print(f"[INFO] sec-parser not used: {e}")
            sections["_parsed_elements_count"] = 0

        # Save to S3 if requested
        if save_to_s3:
            try:
                s3_key = save_extracted_sections_to_s3(
                    ticker, sections, portfolio_name, filing_type
                )
                sections["_saved_path"] = s3_key
                sections["_storage_location"] = "S3"
                sections["_extraction_success"] = True
            except Exception as e:
                print(f"[WARNING] Could not save to S3: {e}")
                sections["_extraction_success"] = False
                sections["_error"] = str(e)
        else:
            sections["_extraction_success"] = True

        # Add extraction summary
        sections["_summary"] = {
            "ticker": ticker,
            "filing_type": filing_type,
            "total_chars": len(sections.get("full_text", "")),
            "sections_found": [
                k for k in sections.keys() if sections.get(k) and not k.startswith("_")
            ],
        }

        return sections

    except Exception as e:
        print(f"[ERROR] Failed to extract filing for {ticker}: {e}")
        import traceback

        traceback.print_exc()
        return {"_extraction_success": False, "_error": str(e)}


# Keep backward compatibility
def extract_key_filing_sections(
    ticker: str,
    html_content: str,
    portfolio_name: str = "default",
    filing_type: str = "10-K",
    save_to_s3: bool = True,
    is_xml: bool = False,
) -> Dict[str, str]:
    """
    Alias for extract_filing_from_html for backward compatibility.

    Args:
        ticker: Company ticker symbol
        html_content: HTML/XML content of the SEC filing
        portfolio_name: Portfolio name to organize files in S3 (default: "default")
        filing_type: Type of filing (10-K, 10-Q, etc.)
        save_to_s3: If True, save extracted data to S3
        is_xml: If True, parse as XML instead of HTML

    Returns:
        Dictionary with extracted sections and metadata
    """
    return extract_filing_from_html(
        ticker, html_content, portfolio_name, filing_type, save_to_s3, is_xml
    )
