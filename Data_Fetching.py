import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

companies = {
    "Reliance": "RELIANCE.NS",
    "Adani": "ADANIENT.NS",
    "Mahindra_Mahindra": "M&M.NS",
    "Maruti_Suzuki": "MARUTI.NS",
    "Nestle_India": "NESTLEIND.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS"
    
}

# Directory to save CSVs
output_dir = "stock_data"
os.makedirs(output_dir, exist_ok=True)

# Fetch data and save CSV
for company_name, ticker in companies.items():
    print(f"Fetching data for {company_name} ({ticker})...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 6)
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Reset index to have 'Date' as a column
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df.sort_values("Date", inplace=True)
        df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        file_path = os.path.join(output_dir, f"{company_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")
    else:
        print(f"⚠️ No data fetched for {company_name}")

print("All done!")
