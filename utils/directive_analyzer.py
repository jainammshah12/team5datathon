"""
Directive Analyzer for Regulatory Documents
Extracts key information from directives that affect stock prices
Handles multiple languages and formats (HTML/XML)
"""

import os
import json
import re
import glob
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

# Language detection
try:
    from langdetect import detect, detect_langs, LangDetectException

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[WARNING] langdetect not available. Install with: pip install langdetect")

# Translation support (optional)
try:
    from deep_translator import GoogleTranslator

    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("[WARNING] deep-translator not available. Translation disabled.")

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


def safe_print(message: str) -> None:
    """Safely print Unicode strings to avoid encoding errors."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Encode to ASCII with error replacement
        safe_message = message.encode("ascii", "replace").decode("ascii")
        print(safe_message)


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of text content.

    Args:
        text: Text content to analyze (first 1000 chars for speed)

    Returns:
        Tuple of (language_code, confidence) e.g., ('en', 0.95)
    """
    if not LANGDETECT_AVAILABLE:
        # Fallback: simple heuristic based on character sets
        return detect_language_heuristic(text)

    try:
        # Use first 1000 characters for faster detection
        sample_text = text[:1000] if len(text) > 1000 else text
        if not sample_text.strip():
            return ("unknown", 0.0)

        # Get language with confidence
        detected_langs = detect_langs(sample_text)
        if detected_langs:
            primary_lang = detected_langs[0]
            return (primary_lang.lang, primary_lang.prob)
        else:
            detected = detect(sample_text)
            return (detected, 0.8)  # Default confidence if prob not available
    except LangDetectException:
        return detect_language_heuristic(text)
    except Exception as e:
        print(f"[WARNING] Language detection failed: {e}")
        return detect_language_heuristic(text)


def detect_language_heuristic(text: str) -> Tuple[str, float]:
    """
    Fallback language detection using character set heuristics.

    Args:
        text: Text content to analyze

    Returns:
        Tuple of (language_code, confidence)
    """
    if not text.strip():
        return ("unknown", 0.0)

    sample = text[:500]

    # Chinese characters (CJK unified)
    if re.search(r"[\u4e00-\u9fff]", sample):
        return ("zh", 0.9)

    # Japanese characters (Hiragana, Katakana, Kanji)
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample):
        return ("ja", 0.9)

    # Korean characters
    if re.search(r"[\uac00-\ud7a3]", sample):
        return ("ko", 0.9)

    # Arabic characters
    if re.search(r"[\u0600-\u06ff]", sample):
        return ("ar", 0.9)

    # Cyrillic characters
    if re.search(r"[\u0400-\u04ff]", sample):
        return ("ru", 0.8)

    # Common European languages - check for French, German, Spanish patterns
    french_words = [
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "et",
        "un",
        "une",
        "par",
        "pour",
        "dans",
        "sur",
    ]
    german_words = ["der", "die", "das", "und", "ist", "sind", "für", "von"]
    spanish_words = ["el", "la", "los", "las", "de", "del", "y", "en", "es", "son"]

    sample_lower = sample.lower()
    french_count = sum(1 for word in french_words if word in sample_lower)
    german_count = sum(1 for word in german_words if word in sample_lower)
    spanish_count = sum(1 for word in spanish_words if word in sample_lower)

    if french_count > 3:
        return ("fr", 0.75)
    if german_count > 3:
        return ("de", 0.75)
    if spanish_count > 3:
        return ("es", 0.75)

    # Default to English
    return ("en", 0.7)


def is_xml_content(content: str) -> bool:
    """
    Detect if content is XML format.

    Args:
        content: File content string

    Returns:
        True if content appears to be XML
    """
    content_stripped = content.strip()
    return (
        content_stripped.startswith("<?xml")
        or content_stripped.startswith("<XML>")
        or "<XML>" in content_stripped[:500]
        or content_stripped.startswith("<bill")
        or content_stripped.startswith("<publicLaw")
    )


def extract_full_text_from_html(html_content: str, is_xml: bool = False) -> str:
    """
    Extract full text content from HTML or XML using BeautifulSoup.

    Args:
        html_content: HTML/XML content of the directive
        is_xml: If True, use XML parser to maintain case sensitivity

    Returns:
        Full text content
    """
    try:
        parser = "xml" if is_xml else "lxml"
        soup = BeautifulSoup(html_content, parser)

        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
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


def extract_dates(text: str, language: str = "en") -> List[Dict[str, str]]:
    """
    Extract dates from directive text (effective dates, deadlines, etc.)

    Args:
        text: Directive text content
        language: Language code for context

    Returns:
        List of extracted dates with context
    """
    dates = []

    # Common date patterns across languages
    # ISO format: YYYY-MM-DD
    iso_pattern = r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"
    for match in re.finditer(iso_pattern, text):
        dates.append(
            {
                "date": f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}",
                "context": text[
                    max(0, match.start() - 50) : min(len(text), match.end() + 50)
                ],
                "type": "effective_date",
            }
        )

    # European format: DD.MM.YYYY or DD/MM/YYYY
    eu_pattern = r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"
    for match in re.finditer(eu_pattern, text):
        dates.append(
            {
                "date": f"{match.group(3)}-{match.group(2).zfill(2)}-{match.group(1).zfill(2)}",
                "context": text[
                    max(0, match.start() - 50) : min(len(text), match.end() + 50)
                ],
                "type": "date",
            }
        )

    # US format: MM/DD/YYYY
    us_pattern = r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"
    for match in re.finditer(us_pattern, text):
        dates.append(
            {
                "date": f"{match.group(3)}-{match.group(1).zfill(2)}-{match.group(2).zfill(2)}",
                "context": text[
                    max(0, match.start() - 50) : min(len(text), match.end() + 50)
                ],
                "type": "date",
            }
        )

    # Year-only patterns (often used in legal texts)
    year_pattern = r"\b(19|20)\d{2}\b"
    for match in re.finditer(year_pattern, text):
        context = text[max(0, match.start() - 100) : min(len(text), match.end() + 100)]
        # Only add if context suggests it's a significant date
        if any(
            word in context.lower()
            for word in [
                "effective",
                "enact",
                "pass",
                "date",
                "year",
                "生效",
                "施行",
                "発効",
            ]
        ):
            dates.append({"date": match.group(0), "context": context, "type": "year"})

    # Remove duplicates
    seen = set()
    unique_dates = []
    for d in dates:
        key = (d["date"], d.get("context", "")[:50])
        if key not in seen:
            seen.add(key)
            unique_dates.append(d)

    return unique_dates[:20]  # Limit to 20 most relevant dates


def extract_financial_impacts(text: str, language: str = "en") -> List[Dict[str, str]]:
    """
    Extract financial impacts: penalties, fines, funding amounts, costs, etc.

    Args:
        text: Directive text content
        language: Language code

    Returns:
        List of financial impacts
    """
    impacts = []

    # Currency patterns (USD, EUR, CNY, JPY, etc.)
    currency_patterns = {
        "en": [
            r"\$\s*([\d,]+(?:\.[\d]+)?)\s*(?:million|billion|trillion|M|B|T)?",
            r"([\d,]+(?:\.[\d]+)?)\s*(?:million|billion|trillion|M|B|T)?\s*(?:USD|dollars?)",
            r"([\d,]+(?:\.[\d]+)?)\s*(?:million|billion|trillion|M|B|T)?\s*(?:EUR|euros?)",
        ],
        "fr": [
            r"([\d,]+(?:\.[\d]+)?)\s*(?:million|milliard|millions|milliards|M|Md)?\s*(?:EUR|euros?)",
            r"€\s*([\d,]+(?:\.[\d]+)?)\s*(?:million|milliard|millions|milliards|M|Md)?",
        ],
        "zh": [
            r"([\d,]+(?:\.[\d]+)?)\s*(?:万元|亿元|万元人民币|亿元人民币|美元|欧元)",
            r"[￥$€]\s*([\d,]+(?:\.[\d]+)?)\s*(?:万元|亿元)?",
        ],
        "ja": [
            r"([\d,]+(?:\.[\d]+)?)\s*(?:万円|億円|百万円|円)",
            r"[￥$€]\s*([\d,]+(?:\.[\d]+)?)\s*(?:万円|億円)?",
        ],
    }

    patterns = currency_patterns.get(language, currency_patterns["en"])

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            context = text[
                max(0, match.start() - 100) : min(len(text), match.end() + 100)
            ]
            # Check if context suggests penalty, fine, funding, cost, etc.
            penalty_keywords = {
                "en": [
                    "penalty",
                    "fine",
                    "sanction",
                    "violation",
                    "non-compliance",
                    "funding",
                    "cost",
                    "tax",
                ],
                "fr": [
                    "amende",
                    "sanction",
                    "violation",
                    "financement",
                    "coût",
                    "taxe",
                    "pénalité",
                ],
                "zh": ["罚款", "处罚", "资金", "成本", "税收", "税", "费用"],
                "ja": ["罰金", "制裁", "資金", "費用", "税金"],
            }

            keywords = penalty_keywords.get(language, penalty_keywords["en"])
            if any(keyword in context.lower() for keyword in keywords):
                impacts.append(
                    {
                        "amount": match.group(1) if match.groups() else match.group(0),
                        "currency": extract_currency_from_context(context, language),
                        "context": context,
                        "type": "financial_impact",
                    }
                )

    return impacts[:30]  # Limit to 30 most relevant impacts


def extract_currency_from_context(context: str, language: str) -> str:
    """Extract currency type from context."""
    context_lower = context.lower()

    if "$" in context or "usd" in context_lower or "dollar" in context_lower:
        return "USD"
    if "€" in context or "eur" in context_lower or "euro" in context_lower:
        return "EUR"
    if "£" in context or "gbp" in context_lower or "pound" in context_lower:
        return "GBP"
    if "¥" in context or "yuan" in context_lower or "人民币" in context:
        return "CNY"
    if "円" in context or "yen" in context_lower:
        return "JPY"

    return "USD"  # Default


def extract_affected_sectors(text: str, language: str = "en") -> List[str]:
    """
    Extract affected sectors/industries from directive text.

    Args:
        text: Directive text content
        language: Language code

    Returns:
        List of affected sectors
    """
    sectors = []

    # Sector keywords in multiple languages
    sector_keywords = {
        "en": {
            "energy": [
                "energy",
                "power",
                "electricity",
                "renewable",
                "solar",
                "wind",
                "nuclear",
                "coal",
                "oil",
                "gas",
            ],
            "technology": [
                "technology",
                "artificial intelligence",
                "ai",
                "machine learning",
                "data",
                "software",
                "digital",
                "cyber",
            ],
            "finance": [
                "financial",
                "banking",
                "investment",
                "securities",
                "trading",
                "market",
            ],
            "healthcare": [
                "healthcare",
                "medical",
                "pharmaceutical",
                "drug",
                "medicine",
                "health",
            ],
            "telecommunications": [
                "telecommunication",
                "telecom",
                "communication",
                "network",
                "internet",
                "5g",
            ],
            "automotive": ["automotive", "vehicle", "car", "truck", "transportation"],
            "manufacturing": ["manufacturing", "production", "factory", "industrial"],
            "consumer": ["consumer", "retail", "e-commerce", "shopping"],
            "environmental": [
                "environmental",
                "emission",
                "carbon",
                "climate",
                "pollution",
                "green",
            ],
            "agriculture": ["agriculture", "farming", "food", "crop"],
        },
        "fr": {
            "energy": [
                "énergie",
                "électricité",
                "renouvelable",
                "solaire",
                "éolien",
                "nucléaire",
                "charbon",
                "pétrole",
                "gaz",
            ],
            "technology": [
                "technologie",
                "intelligence artificielle",
                "ia",
                "données",
                "numérique",
                "cyber",
            ],
            "finance": [
                "financier",
                "banque",
                "investissement",
                "sécurité",
                "trading",
                "marché",
            ],
            "consumer": ["consommateur", "commerce", "e-commerce", "achat"],
        },
        "zh": {
            "energy": [
                "能源",
                "电力",
                "可再生能源",
                "太阳能",
                "风能",
                "核能",
                "煤炭",
                "石油",
                "天然气",
            ],
            "technology": ["技术", "人工智能", "AI", "数据", "软件", "数字", "网络"],
            "finance": ["金融", "银行", "投资", "证券", "交易", "市场"],
            "healthcare": ["医疗", "医药", "药品", "健康"],
            "consumer": ["消费者", "零售", "电子商务", "购物"],
        },
        "ja": {
            "energy": [
                "エネルギー",
                "電力",
                "再生可能エネルギー",
                "太陽光",
                "風力",
                "原子力",
                "石炭",
                "石油",
                "ガス",
            ],
            "technology": [
                "技術",
                "人工知能",
                "AI",
                "データ",
                "ソフトウェア",
                "デジタル",
            ],
            "finance": ["金融", "銀行", "投資", "証券", "取引", "市場"],
            "consumer": ["消費者", "小売", "eコマース"],
        },
    }

    keywords = sector_keywords.get(language, sector_keywords["en"])
    text_lower = text.lower()

    for sector, terms in keywords.items():
        if any(term in text_lower for term in terms):
            if sector not in sectors:
                sectors.append(sector)

    return sectors


def extract_compliance_requirements(text: str, language: str = "en") -> List[str]:
    """
    Extract compliance requirements and obligations.

    Args:
        text: Directive text content
        language: Language code

    Returns:
        List of compliance requirements
    """
    requirements = []

    # Keywords that indicate requirements
    requirement_keywords = {
        "en": [
            "shall",
            "must",
            "required",
            "mandatory",
            "obligation",
            "compliance",
            "implement",
            "enforce",
        ],
        "fr": [
            "doit",
            "nécessaire",
            "obligatoire",
            "obligation",
            "conformité",
            "mettre en œuvre",
            "appliquer",
        ],
        "zh": ["应当", "必须", "要求", "义务", "合规", "实施", "执行"],
        "ja": [
            "しなければならない",
            "必要",
            "義務",
            "コンプライアンス",
            "実施",
            "執行",
        ],
    }

    keywords = requirement_keywords.get(language, requirement_keywords["en"])
    text_lower = text.lower()

    # Extract sentences containing requirement keywords
    sentences = re.split(r"[.!?。！？]\s*", text)
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            cleaned = sentence.strip()
            if len(cleaned) > 20 and len(cleaned) < 500:  # Reasonable length
                requirements.append(cleaned)

    return requirements[:50]  # Limit to 50 requirements


def extract_geographic_scope(text: str, language: str = "en") -> List[str]:
    """
    Extract geographic scope/jurisdiction.

    Args:
        text: Directive text content
        language: Language code

    Returns:
        List of geographic regions/countries
    """
    regions = []

    # Common regions and countries
    regions_keywords = {
        "en": [
            "United States",
            "US",
            "USA",
            "European Union",
            "EU",
            "China",
            "Japan",
            "United Kingdom",
            "UK",
            "global",
            "international",
            "national",
            "regional",
            "state",
            "federal",
        ],
        "fr": [
            "États-Unis",
            "États Unis",
            "UE",
            "Union européenne",
            "Chine",
            "Japon",
            "Royaume-Uni",
            "mondial",
            "international",
            "national",
            "régional",
            "fédéral",
        ],
        "zh": [
            "美国",
            "欧盟",
            "欧洲",
            "日本",
            "英国",
            "全球",
            "国际",
            "国家",
            "地区",
            "联邦",
        ],
        "ja": [
            "米国",
            "アメリカ",
            "EU",
            "欧州連合",
            "中国",
            "英国",
            "グローバル",
            "国際",
            "国内",
            "地域",
        ],
    }

    keywords = regions_keywords.get(language, regions_keywords["en"])
    text_lower = text.lower()

    for keyword in keywords:
        if keyword.lower() in text_lower:
            if keyword not in regions:
                regions.append(keyword)

    return regions


def extract_sections_from_directive(
    html_content: str, is_xml: bool = False, language: str = "en"
) -> Dict[str, str]:
    """
    Extract specific sections from directive HTML/XML.

    Args:
        html_content: HTML/XML content of the directive
        is_xml: If True, parse as XML
        language: Language code for context

    Returns:
        Dictionary with extracted sections
    """
    sections = {
        "title": "",
        "effective_date": "",
        "geographic_scope": "",
        "affected_sectors": "",
        "key_provisions": "",
        "compliance_requirements": "",
        "penalties_sanctions": "",
        "financial_impacts": "",
        "implementation_deadlines": "",
        "amendments_references": "",
        "full_text": "",
    }

    try:
        # Detect XML if not specified
        if not is_xml and is_xml_content(html_content):
            is_xml = True
            print("[INFO] Detected XML format, using XML parser")

        # Extract full text
        sections["full_text"] = extract_full_text_from_html(html_content, is_xml=is_xml)

        if not sections["full_text"]:
            print(f"[WARNING] No text extracted from directive")
            return sections

        text = sections["full_text"]
        print(f"[INFO] Extracted full text: {len(text)} characters")

        # Extract title (usually first 200 characters or first heading)
        soup = BeautifulSoup(html_content, "xml" if is_xml else "lxml")
        title_tag = soup.find(["title", "h1", "h2"])
        if title_tag:
            sections["title"] = title_tag.get_text(strip=True)
        else:
            sections["title"] = text[:200].split("\n")[0].strip()

        # Extract dates
        dates = extract_dates(text, language)
        if dates:
            sections["effective_date"] = "\n".join(
                [f"{d['date']}: {d['context'][:100]}..." for d in dates[:10]]
            )
            sections["implementation_deadlines"] = sections["effective_date"]

        # Extract geographic scope
        regions = extract_geographic_scope(text, language)
        if regions:
            sections["geographic_scope"] = ", ".join(regions)

        # Extract affected sectors
        sectors = extract_affected_sectors(text, language)
        if sectors:
            sections["affected_sectors"] = ", ".join(sectors)

        # Extract compliance requirements
        requirements = extract_compliance_requirements(text, language)
        if requirements:
            sections["compliance_requirements"] = "\n\n".join(requirements[:20])

        # Extract financial impacts
        financial = extract_financial_impacts(text, language)
        if financial:
            sections["financial_impacts"] = "\n".join(
                [
                    f"{f.get('amount', 'N/A')} {f.get('currency', '')}: {f['context'][:150]}..."
                    for f in financial[:15]
                ]
            )

        # Extract key provisions (abstract/summary - first 1000 chars or preamble)
        # Look for common section headers
        provision_keywords = {
            "en": [
                "preamble",
                "summary",
                "purpose",
                "objective",
                "scope",
                "article",
                "section",
                "chapter",
            ],
            "fr": [
                "préambule",
                "résumé",
                "objet",
                "objectif",
                "portée",
                "article",
                "section",
                "chapitre",
            ],
            "zh": ["序言", "摘要", "目的", "目标", "范围", "条", "章", "节"],
            "ja": ["前文", "要約", "目的", "対象", "範囲", "条", "章", "節"],
        }

        keywords = provision_keywords.get(language, provision_keywords["en"])
        lines = text.split("\n")
        provisions_start = 0
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                provisions_start = i
                break

        if provisions_start > 0:
            sections["key_provisions"] = "\n".join(
                lines[provisions_start : provisions_start + 50]
            )
        else:
            sections["key_provisions"] = text[:1500]

        # Extract penalties and sanctions section
        penalty_keywords = {
            "en": ["penalty", "fine", "sanction", "violation", "non-compliance"],
            "fr": ["amende", "sanction", "violation", "non-conformité"],
            "zh": ["罚款", "处罚", "制裁", "违规", "不合规"],
            "ja": ["罰金", "制裁", "違反", "不遵守"],
        }

        keywords = penalty_keywords.get(language, penalty_keywords["en"])
        penalty_lines = [
            line
            for line in lines
            if any(keyword in line.lower() for keyword in keywords)
        ]
        if penalty_lines:
            sections["penalties_sanctions"] = "\n".join(penalty_lines[:30])

        # Extract amendments and references
        reference_keywords = {
            "en": [
                "amend",
                "modify",
                "reference",
                "cite",
                "directive",
                "regulation",
                "law",
            ],
            "fr": ["modifier", "référence", "citer", "directive", "règlement", "loi"],
            "zh": ["修改", "修正", "参考", "引用", "指令", "法规", "法律"],
            "ja": ["改正", "修正", "参照", "引用", "指令", "規制", "法律"],
        }

        keywords = reference_keywords.get(language, reference_keywords["en"])
        ref_lines = [
            line
            for line in lines
            if any(keyword in line.lower() for keyword in keywords)
        ]
        if ref_lines:
            sections["amendments_references"] = "\n".join(ref_lines[:30])

        # Print statistics
        for section_name, content in sections.items():
            if section_name != "full_text" and content:
                word_count = len(content.split())
                print(f"[INFO] Extracted {section_name}: {word_count} words")

    except Exception as e:
        print(f"[ERROR] Failed to extract sections: {e}")
        import traceback

        traceback.print_exc()

    return sections


def check_existing_directive_extraction(
    directive_name: str, portfolio_name: str = "default"
) -> Optional[Tuple[str, Dict]]:
    """
    Check if directive extraction already exists in S3.

    Args:
        directive_name: Name/identifier of the directive
        portfolio_name: Portfolio name for organization

    Returns:
        Tuple of (s3_key, extraction_dict) if found, None otherwise
    """
    if not S3_AVAILABLE:
        return None

    try:
        # Clean filename for matching
        safe_name = re.sub(r"[^\w\s-]", "", directive_name)[:50]
        s3_prefix = f"data/extracted_directives/{portfolio_name}/"

        # List files in S3
        existing_files = list_files_in_s3(s3_prefix)

        # Find matching files (both complete.json and nlp_extraction.json)
        matching_files = [
            f for f in existing_files if safe_name in f and f.endswith((".json"))
        ]

        if matching_files:
            # Use the most recent one (last in sorted list by timestamp)
            latest_file = sorted(matching_files)[-1]
            print(f"[INFO] Found existing extraction in S3: {latest_file}")

            # Download and parse
            from .s3_utils import read_file_from_s3

            content = read_file_from_s3(latest_file)
            extraction = json.loads(content)

            return (latest_file, extraction)
        else:
            print(f"[INFO] No existing extraction found for: {directive_name}")
            return None

    except Exception as e:
        print(f"[WARNING] Could not check for existing extraction: {e}")
        return None


def save_extracted_directive_to_s3(
    directive_name: str,
    sections: Dict[str, str],
    language: str,
    portfolio_name: str = "default",
) -> str:
    """
    Save extracted directive sections to S3 as JSON.

    Args:
        directive_name: Name/identifier of the directive
        sections: Dictionary with extracted sections
        language: Detected language code
        portfolio_name: Portfolio name for organization

    Returns:
        S3 key (path) of saved JSON file
    """
    if not S3_AVAILABLE:
        raise ConnectionError("S3 utilities not available. Cannot save to S3.")

    # Clean up old extraction files for this directive in S3
    s3_prefix = f"data/extracted_directives/{portfolio_name}/"

    try:
        existing_files = list_files_in_s3(s3_prefix)
        # Filter for files matching this directive (use sanitized name for matching)
        safe_name_check = re.sub(r"[^\w\s-]", "", directive_name)[:50]
        matching_files = [
            f for f in existing_files if safe_name_check in f and f.endswith(".json")
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

    # Prepare output data
    output_data = {
        "metadata": {
            "directive_name": directive_name,
            "language": language,
            "extraction_date": datetime.now().isoformat(),
            "extractor_version": "1.0",
        },
        "sections": {
            "title": sections.get("title", ""),
            "effective_date": sections.get("effective_date", ""),
            "geographic_scope": sections.get("geographic_scope", ""),
            "affected_sectors": sections.get("affected_sectors", ""),
            "key_provisions": sections.get("key_provisions", ""),
            "compliance_requirements": sections.get("compliance_requirements", ""),
            "penalties_sanctions": sections.get("penalties_sanctions", ""),
            "financial_impacts": sections.get("financial_impacts", ""),
            "implementation_deadlines": sections.get("implementation_deadlines", ""),
            "amendments_references": sections.get("amendments_references", ""),
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
        "stock_impact_focus": {
            "description": "Sections optimized for stock price impact analysis",
            "high_impact_indicators": [
                "financial_impacts - direct cost implications for companies",
                "penalties_sanctions - compliance risk and potential fines",
                "compliance_requirements - operational cost and business model changes",
                "affected_sectors - identifies which industries are impacted",
                "effective_date - timing of regulatory changes affecting market",
                "geographic_scope - market reach and jurisdiction impact",
            ],
        },
    }

    # Upload to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\s-]", "", directive_name)[:50]  # Sanitize filename
    filename = f"{safe_name}_{timestamp}_complete.json"
    s3_key = f"data/extracted_directives/{portfolio_name}/{filename}"

    # Convert to JSON string and encode to bytes
    json_content = json.dumps(output_data, indent=2, ensure_ascii=False)
    json_bytes = json_content.encode("utf-8")

    # Upload to S3
    try:
        upload_file_to_s3(json_bytes, s3_key, overwrite=False)
        file_size_kb = len(json_bytes) / 1024
        print(f"[INFO] Saved complete directive to S3: {s3_key}")
        print(f"[INFO] File size: {file_size_kb:.2f} KB")
        print(f"[INFO] Language: {language}")
        print(
            f"[INFO] Sections extracted: {', '.join(output_data['statistics']['sections_found'])}"
        )
        return s3_key
    except Exception as e:
        print(f"[ERROR] Failed to upload to S3: {e}")
        raise


def extract_directive_from_file(
    file_path: str,
    portfolio_name: str = "default",
    save_to_s3: bool = True,
    force_reprocess: bool = False,
) -> Dict[str, str]:
    """
    Extract key information from a directive file (HTML/XML).

    Args:
        file_path: Path to directive file (HTML or XML)
        portfolio_name: Portfolio name for organization
        save_to_s3: If True, save extracted data to S3
        force_reprocess: If True, reprocess even if extraction exists in S3

    Returns:
        Dictionary with extracted sections and metadata
    """
    try:
        # Get directive name from file path
        directive_name = (
            os.path.basename(file_path).replace(".html", "").replace(".xml", "")
        )

        # Check if extraction already exists in S3 (lazy loading)
        if not force_reprocess and S3_AVAILABLE:
            existing = check_existing_directive_extraction(
                directive_name, portfolio_name
            )
            if existing:
                s3_key, extraction_dict = existing
                print(f"[INFO] Using existing extraction from S3: {s3_key}")

                # Convert S3 format to expected format
                if "sections" in extraction_dict:
                    sections = extraction_dict["sections"]
                elif "basic_extraction" in extraction_dict:
                    sections = extraction_dict["basic_extraction"]
                else:
                    sections = extraction_dict

                # Add metadata
                sections["_language"] = extraction_dict.get("metadata", {}).get(
                    "language", "unknown"
                )
                sections["_language_confidence"] = extraction_dict.get(
                    "metadata", {}
                ).get("language_confidence", 0.0)
                sections["_is_xml"] = extraction_dict.get("metadata", {}).get(
                    "is_xml", False
                )
                sections["_saved_path"] = s3_key
                sections["_storage_location"] = "S3"
                sections["_extraction_success"] = True
                sections["_cached"] = True

                return sections

        # Read file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Directive file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        safe_print(f"[INFO] Processing directive file: {file_path}")
        print(f"[INFO] Content size: {len(content)} characters")

        # Detect format
        is_xml = is_xml_content(content)

        # Extract text
        text = extract_full_text_from_html(content, is_xml=is_xml)

        if not text:
            print(f"[WARNING] No text extracted from directive")
            return {"_extraction_success": False, "_error": "No text extracted"}

        # Detect language
        language, confidence = detect_language(text)
        print(f"[INFO] Detected language: {language} (confidence: {confidence:.2f})")

        # Extract sections
        sections = extract_sections_from_directive(
            content, is_xml=is_xml, language=language
        )

        # Add metadata
        sections["_language"] = language
        sections["_language_confidence"] = confidence
        sections["_is_xml"] = is_xml

        # Save to S3 if requested
        if save_to_s3:
            try:
                directive_name = (
                    os.path.basename(file_path).replace(".html", "").replace(".xml", "")
                )
                s3_key = save_extracted_directive_to_s3(
                    directive_name, sections, language, portfolio_name
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

        # Add summary
        sections["_summary"] = {
            "file_path": file_path,
            "language": language,
            "total_chars": len(sections.get("full_text", "")),
            "sections_found": [
                k for k in sections.keys() if sections.get(k) and not k.startswith("_")
            ],
        }

        return sections

    except Exception as e:
        safe_print(f"[ERROR] Failed to extract directive from {file_path}: {e}")
        import traceback

        traceback.print_exc()
        return {"_extraction_success": False, "_error": str(e)}


def extract_directives_from_directory(
    directory_path: str, portfolio_name: str = "default", save_to_s3: bool = True
) -> List[Dict[str, str]]:
    """
    Extract information from all directive files in a directory.

    Args:
        directory_path: Path to directory containing directive files
        portfolio_name: Portfolio name for organization
        save_to_s3: If True, save extracted data to S3

    Returns:
        List of extraction results for each directive
    """
    results = []

    if not os.path.exists(directory_path):
        print(f"[ERROR] Directory not found: {directory_path}")
        return results

    # Find all HTML and XML files
    html_files = glob.glob(os.path.join(directory_path, "*.html"))
    xml_files = glob.glob(os.path.join(directory_path, "*.xml"))
    all_files = html_files + xml_files

    print(f"[INFO] Found {len(all_files)} directive files to process")

    for file_path in all_files:
        try:
            result = extract_directive_from_file(file_path, portfolio_name, save_to_s3)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            results.append(
                {
                    "_extraction_success": False,
                    "_file_path": file_path,
                    "_error": str(e),
                }
            )

    return results


def extract_directive_with_nlp(
    file_path: str,
    portfolio_name: str = "default",
    use_nlp_pipeline: bool = True,
    force_reprocess: bool = False,
) -> Dict:
    """
    Extract directive using the enhanced two-layer NLP pipeline.

    This function integrates the basic extraction (text, dates, sectors) with
    the advanced NLP pipeline (entity extraction, impact summarization).

    Args:
        file_path: Path to directive file (HTML or XML)
        portfolio_name: Portfolio name for organization
        use_nlp_pipeline: If True, use the two-layer NLP pipeline for enhanced extraction
        force_reprocess: If True, force reprocessing even if cached extraction exists

    Returns:
        Combined extraction results with both basic and NLP-enhanced data
    """
    try:
        if not use_nlp_pipeline:
            # Use basic extraction only
            return extract_directive_from_file(
                file_path, portfolio_name, save_to_s3=True
            )

        # Use enhanced NLP pipeline
        try:
            from .directive_nlp_pipeline import DirectiveNLPPipeline

            pipeline = DirectiveNLPPipeline(portfolio_name=portfolio_name)
            nlp_result = pipeline.process_directive(
                directive_path=file_path, force_reprocess=force_reprocess
            )

            # Also run basic extraction for compatibility
            basic_result = extract_directive_from_file(
                file_path, portfolio_name, save_to_s3=False
            )

            # Combine results
            combined_result = {
                "basic_extraction": {
                    "title": basic_result.get("title", ""),
                    "effective_date": basic_result.get("effective_date", ""),
                    "geographic_scope": basic_result.get("geographic_scope", ""),
                    "affected_sectors": basic_result.get("affected_sectors", ""),
                    "compliance_requirements": basic_result.get(
                        "compliance_requirements", ""
                    ),
                    "financial_impacts": basic_result.get("financial_impacts", ""),
                    "penalties_sanctions": basic_result.get("penalties_sanctions", ""),
                },
                "nlp_extraction": nlp_result,
                "metadata": {
                    "file_path": file_path,
                    "extraction_method": "combined_basic_nlp",
                    "pipeline_version": nlp_result.get("metadata", {}).get(
                        "pipeline_version", "2.0"
                    ),
                    "extraction_date": nlp_result.get("metadata", {}).get(
                        "extraction_date", datetime.now().isoformat()
                    ),
                    "nlp_services_used": nlp_result.get("metadata", {}).get(
                        "nlp_services_used", {}
                    ),
                },
                "_extraction_success": True,
            }

            safe_print(f"[SUCCESS] Combined extraction completed for {file_path}")
            return combined_result

        except ImportError:
            safe_print(
                "[WARNING] NLP pipeline not available, falling back to basic extraction"
            )
            return extract_directive_from_file(
                file_path, portfolio_name, save_to_s3=True
            )

    except Exception as e:
        safe_print(f"[ERROR] Failed to extract directive with NLP: {e}")
        import traceback

        traceback.print_exc()
        return {"_extraction_success": False, "_file_path": file_path, "_error": str(e)}


def batch_extract_directives_with_nlp(
    directory_path: str,
    portfolio_name: str = "default",
    use_nlp_pipeline: bool = True,
    force_reprocess: bool = False,
) -> Dict[str, Dict]:
    """
    Batch extract directives from directory using NLP pipeline.

    Args:
        directory_path: Path to directory containing directive files
        portfolio_name: Portfolio name for organization
        use_nlp_pipeline: If True, use the two-layer NLP pipeline
        force_reprocess: If True, force reprocessing of all directives

    Returns:
        Dictionary mapping filename -> extraction results
    """
    results = {}

    if not os.path.exists(directory_path):
        print(f"[ERROR] Directory not found: {directory_path}")
        return results

    # Find all HTML and XML files
    html_files = glob.glob(os.path.join(directory_path, "*.html"))
    xml_files = glob.glob(os.path.join(directory_path, "*.xml"))
    all_files = html_files + xml_files

    print(f"[INFO] Found {len(all_files)} directive files to process")
    print(f"[INFO] NLP Pipeline: {'Enabled' if use_nlp_pipeline else 'Disabled'}")
    print(
        f"[INFO] Force Reprocess: {'Yes' if force_reprocess else 'No (lazy loading)'}"
    )

    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            result = extract_directive_with_nlp(
                file_path=file_path,
                portfolio_name=portfolio_name,
                use_nlp_pipeline=use_nlp_pipeline,
                force_reprocess=force_reprocess,
            )
            results[filename] = result

            if result.get("_extraction_success"):
                print(f"[SUCCESS] Processed {filename}")
            else:
                print(f"[ERROR] Failed to process {filename}")

        except Exception as e:
            print(f"[ERROR] Exception processing {filename}: {e}")
            results[filename] = {
                "_extraction_success": False,
                "_file_path": file_path,
                "_error": str(e),
            }

    success_count = sum(1 for r in results.values() if r.get("_extraction_success"))
    print(
        f"\n[INFO] Batch processing complete: {success_count}/{len(results)} successful"
    )

    return results
