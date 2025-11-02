import os
import pandas as pd
from datetime import datetime, timedelta
from utils.yfinance_fetcher import fetch_daily_stock_data, get_sp500_tickers
from utils.s3_utils import upload_file_to_s3
from dotenv import load_dotenv

load_dotenv()


def upload_to_s3_daily_data(csv_file_path: str) -> None:
    """
    Upload CSV file to S3 in the data/daily_data/ subdirectory.

    Args:
        csv_file_path: Path to local CSV file

    Returns:
        None (prints status messages)
    """
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File not found: {csv_file_path}")
        return

    filename = os.path.basename(csv_file_path)
    s3_key = f"data/daily_data/{filename}"

    print(f"\n📤 Uploading to S3: {s3_key}")

    try:
        with open(csv_file_path, "rb") as f:
            file_content = f.read()
            upload_file_to_s3(file_content, s3_key)
            print(f"✅ Successfully uploaded to S3!")
            print(f"   Location: s3://<bucket>/{s3_key}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(
            "   Make sure your .env file has correct AWS credentials and S3_BUCKET_NAME"
        )


def prepare_daily_stock_data(
    tickers: list = None,
    start_date: str = None,
    end_date: str = None,
    output_file: str = None,
    delay: float = 0.1,
    upload_to_s3: bool = False,
) -> pd.DataFrame:
    # Set default dates
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Get tickers if not provided
    if tickers is None:
        print("Loading S&P 500 tickers...")
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} tickers")

    # Fetch daily data for all tickers
    print(f"\nFetching daily stock data from {start_date} to {end_date}...")
    all_data = []

    for i, ticker in enumerate(tickers, 1):
        print(f"Processing {ticker} ({i}/{len(tickers)})...", end="\r")
        data = fetch_daily_stock_data(ticker, start_date, end_date)

        if not data.empty:
            all_data.append(data)

        # Rate limiting
        if delay > 0:
            import time

            time.sleep(delay)

    print()  # New line after progress

    if not all_data:
        print("Error: No data fetched")
        return pd.DataFrame()

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)

    # Clean and prepare data
    # Sort by date and ticker
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Calculate daily performance metrics
    df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()
    df["Price_Change"] = df.groupby("Ticker")["Close"].diff()
    df["Price_Change_Pct"] = df["Price_Change"] / df["Close"].shift(1) * 100
    df["Volume_Change"] = df.groupby("Ticker")["Volume"].diff()

    # Rename columns for consistency (remove spaces)
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Select and order columns for clean output
    column_order = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Daily_Return",
        "Price_Change",
        "Price_Change_Pct",
        "Dividends",
        "Stock_Splits",
    ]
    # Keep only columns that exist
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]

    # Save to CSV
    if output_file is None:
        output_file = f"daily_stock_data_{datetime.now().strftime('%Y-%m-%d')}.csv"

    print(f"\nSaving data to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df)} rows for {df['Ticker'].nunique()} stocks")

    # Upload to S3 if requested
    if upload_to_s3:
        upload_to_s3_daily_data(output_file)

    return df


def query_stock_data(
    ticker: str, start_date: str, end_date: str = None
) -> pd.DataFrame:
    """
    Query daily stock data for a specific ticker and date range.

    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)

    Returns:
        DataFrame with daily stock data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = fetch_daily_stock_data(ticker, start_date, end_date)

    if not data.empty:
        # Calculate metrics
        data["Daily_Return"] = data["Close"].pct_change()
        data["Price_Change"] = data["Close"].diff()
        data["Price_Change_Pct"] = data["Price_Change"] / data["Close"].shift(1) * 100

        # Rename columns
        data.columns = [col.replace(" ", "_") for col in data.columns]

    return data


def query_multiple_stocks(
    tickers: list, start_date: str, end_date: str = None
) -> pd.DataFrame:
    """
    Query daily stock data for multiple tickers.

    Args:
        tickers: List of stock symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)

    Returns:
        DataFrame with data for all specified stocks
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}...")
    all_data = []

    for ticker in tickers:
        data = query_stock_data(ticker, start_date, end_date)
        if not data.empty:
            all_data.append(data)
        import time

        time.sleep(0.1)  # Rate limiting

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return pd.DataFrame()


if __name__ == "__main__":
    """
    Main execution: Prepare daily stock data for all S&P 500 stocks.

    Modify parameters below to customize:
    - Date range
    - Specific tickers (instead of all S&P 500)
    - Output filename
    """
    print("=" * 60)
    print("Daily Stock Data Preparation")
    print("=" * 60)

    # Customize these parameters
    start_date = (datetime.now() - timedelta(days=180)).strftime(
        "%Y-%m-%d"
    )  # Last 6 months
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Option 1: Fetch all S&P 500 stocks (takes time)
    output_file = f"daily_stock_data_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df = prepare_daily_stock_data(
        start_date=start_date,
        end_date=end_date,
        delay=0.1,
        upload_to_s3=True,  # Set to True to automatically upload to S3
        output_file=output_file,
    )

    # Option 2: Fetch specific stocks only (faster for testing)
    # df = prepare_daily_stock_data(
    #     tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    #     start_date=start_date,
    #     end_date=end_date,
    #     output_file='daily_stock_data_sample.csv'
    # )

    # Display summary
    if not df.empty:
        print("\n" + "=" * 60)
        print("Data Summary:")
        print("=" * 60)
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Number of stocks: {df['Ticker'].nunique()}")
        print(f"Total records: {len(df)}")
        print(f"\nColumns: {', '.join(df.columns.tolist())}")
        print(f"\nSample data (first 5 rows):")
        print(df.head())

        print("\n" + "=" * 60)
        print("S3 Upload:")
        print("=" * 60)
        print(
            "To automatically upload to S3, set upload_to_s3=True in prepare_daily_stock_data()"
        )
        print("Or manually upload using:")
        print(f"    from prepare_daily_stock_data import upload_to_s3_daily_data")
        output_filename = (
            output_file
            if "output_file" in locals()
            else "daily_stock_data_YYYY-MM-DD.csv"
        )
        print(f"    upload_to_s3_daily_data('{output_filename}')")
