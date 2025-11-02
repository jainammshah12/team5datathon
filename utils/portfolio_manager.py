"""Portfolio management utilities for handling user portfolios."""

import pandas as pd
import json
from typing import Dict, List, Optional, Tuple


def parse_portfolio_csv(file_content: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse portfolio from CSV file.

    Expected CSV format:
        Ticker,Weight  (or Ticker,Weight%)
        AAPL,0.10
        MSFT,0.08
        GOOGL,0.07

    Args:
        file_content: CSV file content as string

    Returns:
        Tuple of (portfolio_dict, error_message)
        portfolio_dict: {"holdings": [{"ticker": "AAPL", "weight": 0.10}, ...]}
        error_message: None if successful, error string if failed
    """
    try:
        # Try to read CSV
        from io import StringIO

        df = pd.read_csv(StringIO(file_content))

        # Check required columns
        required_cols = ["Ticker", "Weight"]
        if not all(col in df.columns for col in required_cols):
            # Try case-insensitive match
            df.columns = df.columns.str.strip()
            if "ticker" not in df.columns.str.lower():
                return None, "CSV must contain 'Ticker' column"
            if "weight" not in df.columns.str.lower():
                return None, "CSV must contain 'Weight' column"

            # Normalize column names
            df.columns = df.columns.str.lower()
            df = df.rename(columns={"ticker": "Ticker", "weight": "Weight"})

        # Clean and validate data
        df = df[["Ticker", "Weight"]].copy()
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

        # Remove rows with missing values
        df = df.dropna()

        if len(df) == 0:
            return None, "No valid rows found in CSV"

        # Convert weights (handle percentages)
        df["Weight"] = df["Weight"].apply(lambda x: x / 100 if x > 1 else x)

        # Validate weights sum
        total_weight = df["Weight"].sum()
        if total_weight > 1.05 or total_weight < 0.95:  # Allow small rounding errors
            return (
                None,
                f"Weights should sum to 100% (currently {total_weight*100:.1f}%)",
            )

        # Convert to portfolio format
        holdings = df.to_dict("records")
        portfolio = {
            "holdings": [
                {"ticker": row["Ticker"], "weight": float(row["Weight"])}
                for row in holdings
            ]
        }

        return portfolio, None

    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"


def parse_portfolio_manual(portfolio_text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse portfolio from manual text input.

    Expected format (one per line):
        AAPL 10%
        MSFT 8%
        GOOGL 7%

    Or:
        AAPL,0.10
        MSFT,0.08
        GOOGL,0.07

    Args:
        portfolio_text: Text input with ticker and weight

    Returns:
        Tuple of (portfolio_dict, error_message)
    """
    try:
        holdings = []
        lines = portfolio_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try comma-separated format
            if "," in line:
                parts = line.split(",")
                if len(parts) != 2:
                    return (
                        None,
                        f"Invalid format in line: {line}\nExpected: TICKER,WEIGHT",
                    )
                ticker = parts[0].strip().upper()
                weight_str = parts[1].strip()
            else:
                # Try space-separated format
                parts = line.split()
                if len(parts) != 2:
                    return (
                        None,
                        f"Invalid format in line: {line}\nExpected: TICKER WEIGHT or TICKER,WEIGHT",
                    )
                ticker = parts[0].strip().upper()
                weight_str = parts[1].strip()

            # Parse weight (handle %)
            if "%" in weight_str:
                weight = float(weight_str.replace("%", "")) / 100
            else:
                weight = float(weight_str)
                if weight > 1:
                    weight = weight / 100  # Assume percentage if > 1

            holdings.append({"ticker": ticker, "weight": weight})

        if len(holdings) == 0:
            return None, "No portfolio holdings found"

        # Validate weights sum
        total_weight = sum(h["weight"] for h in holdings)
        if total_weight > 1.05 or total_weight < 0.95:
            return (
                None,
                f"Weights should sum to 100% (currently {total_weight*100:.1f}%)",
            )

        portfolio = {"holdings": holdings}
        return portfolio, None

    except Exception as e:
        return None, f"Error parsing portfolio: {str(e)}"


def validate_portfolio(portfolio: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate portfolio structure and data.

    Args:
        portfolio: Portfolio dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(portfolio, dict):
        return False, "Portfolio must be a dictionary"

    if "holdings" not in portfolio:
        return False, "Portfolio must have 'holdings' key"

    holdings = portfolio["holdings"]
    if not isinstance(holdings, list):
        return False, "Holdings must be a list"

    if len(holdings) == 0:
        return False, "Portfolio must have at least one holding"

    total_weight = 0
    for i, holding in enumerate(holdings):
        if not isinstance(holding, dict):
            return False, f"Holding {i+1} must be a dictionary"

        if "ticker" not in holding:
            return False, f"Holding {i+1} missing 'ticker'"

        if "weight" not in holding:
            return False, f"Holding {i+1} missing 'weight'"

        try:
            weight = float(holding["weight"])
            if weight < 0 or weight > 1:
                return False, f"Holding {i+1} weight must be between 0 and 1 (0-100%)"
            total_weight += weight
        except (ValueError, TypeError):
            return False, f"Holding {i+1} weight must be a number"

    if abs(total_weight - 1.0) > 0.05:  # Allow 5% tolerance
        return (
            False,
            f"Total portfolio weight should be 100% (currently {total_weight*100:.1f}%)",
        )

    return True, None


def portfolio_to_dataframe(portfolio: Dict) -> pd.DataFrame:
    """
    Convert portfolio to DataFrame for display.

    Args:
        portfolio: Portfolio dictionary

    Returns:
        DataFrame with columns: Ticker, Weight, Weight%
    """
    holdings = portfolio.get("holdings", [])
    df = pd.DataFrame(holdings)
    if len(df) > 0:
        df["Weight%"] = (df["weight"] * 100).round(2)
        df = df[["ticker", "weight", "Weight%"]].copy()
        df.columns = ["Ticker", "Weight", "Weight %"]
        df = df.sort_values("Weight", ascending=False)
    return df


def get_portfolio_summary(portfolio: Dict) -> str:
    """
    Get a formatted summary of the portfolio.

    Args:
        portfolio: Portfolio dictionary

    Returns:
        Formatted markdown string
    """
    holdings = portfolio.get("holdings", [])
    if len(holdings) == 0:
        return "**Portfolio is empty**"

    total_weight = sum(h["weight"] for h in holdings)

    summary = f"## Portfolio Summary\n\n"
    summary += f"**Number of Holdings:** {len(holdings)}\n"
    summary += f"**Total Weight:** {total_weight*100:.2f}%\n\n"

    summary += "### Holdings:\n\n"
    for holding in sorted(holdings, key=lambda x: x["weight"], reverse=True):
        summary += f"- **{holding['ticker']}**: {holding['weight']*100:.2f}%\n"

    return summary
