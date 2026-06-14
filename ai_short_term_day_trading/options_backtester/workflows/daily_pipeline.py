from prefect import task, flow, get_run_logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import time
from dotenv import load_dotenv

# 載入 .env 變數
load_dotenv()

# Add parent directory to path to allow importing TimescaleDBClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.timescale_client import TimescaleDBClient

# Import DayTradingDataEngine from ai_short_term_day_trading
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_engine import DayTradingDataEngine

@task(retries=3, retry_delay_seconds=60)
def fetch_shioaji_data(target_date: str) -> pd.DataFrame:
    """
    Fetch Options Tick and Order Book data via Shioaji.
    Uses DayTradingDataEngine to login and fetch real option contracts.
    """
    logger = get_run_logger()
    logger.info(f"Fetching Shioaji Options data for {target_date}")
    
    engine = DayTradingDataEngine()
    
    all_dfs = []
    
    try:
        # 找出幾個熱門的合約 (示範: 找出 Call 和 Put 各前 3 熱門的)
        call_contract = engine.get_best_volume_option_contract('Call', allocated_capital=150000)
        put_contract = engine.get_best_volume_option_contract('Put', allocated_capital=150000)
        
        target_contracts = []
        if call_contract: target_contracts.append(call_contract)
        if put_contract: target_contracts.append(put_contract)
        
        if not target_contracts:
            logger.warning("No active option contracts found!")
            return pd.DataFrame()
            
        for contract in target_contracts:
            logger.info(f"Downloading ticks for: {contract.symbol}")
            ticks = engine.api.ticks(contract, target_date)
            
            if not ticks or not ticks.ts:
                logger.warning(f"No tick data found for {contract.symbol} on {target_date}")
                continue
                
            df = pd.DataFrame({**ticks})
            if df.empty:
                continue
                
            df['time'] = pd.to_datetime(df['ts']).dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
            df['symbol'] = contract.symbol
            
            df['strike'] = float(getattr(contract, 'strike_price', 0.0))
            opt_type = getattr(contract, 'option_right', 'Call')
            df['option_type'] = opt_type.name if hasattr(opt_type, 'name') else str(opt_type)
            
            delivery_date = getattr(contract, 'delivery_date', "2026/01/01")
            df['expiry_date'] = pd.to_datetime(delivery_date).date()
            
            df.rename(columns={'close': 'price'}, inplace=True)
            
            df = df[['time', 'symbol', 'strike', 'option_type', 'expiry_date', 
                     'price', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume']]
            
            all_dfs.append(df)
            time.sleep(1) # Rate limit protection
            
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df
    else:
        logger.warning("Returning empty DataFrame.")
        return pd.DataFrame(columns=['time', 'symbol', 'strike', 'option_type', 'expiry_date', 
                                     'price', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume'])



@task
def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Cleaning and Missing Value Imputation using Forward Fill.
    """
    logger = get_run_logger()
    logger.info("Cleaning data and applying Forward Fill imputation")
    
    # Sort by time and symbol
    df = df.sort_values(['symbol', 'time'])
    
    # Forward fill missing prices/volumes
    df[['price', 'bid_price', 'ask_price']] = df.groupby('symbol')[['price', 'bid_price', 'ask_price']].ffill()
    df[['price', 'bid_price', 'ask_price']] = df[['price', 'bid_price', 'ask_price']].bfill() # bfill if start has NaNs
    
    return df

@task
def calculate_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Implied Volatility (IV) for the dataset.
    This acts as a placeholder for the Vectorized BS Engine computation.
    """
    logger = get_run_logger()
    logger.info("Calculating IV Surface")
    
    # Mock IV calculation
    # In reality, this will use core/bs_vectorized.py to compute IV
    df['implied_volatility'] = np.random.uniform(0.15, 0.25, len(df))
    return df

@task
def ingest_to_timescaledb(df: pd.DataFrame):
    """
    Ingest the cleaned and processed DataFrame into TimescaleDB.
    """
    logger = get_run_logger()
    logger.info(f"Ingesting {len(df)} rows into TimescaleDB")
    
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/finance_db")
    client = TimescaleDBClient(db_url)
    
    try:
        client.initialize_schema()
        # Ensure correct column order before insertion
        insert_df = df[['time', 'symbol', 'strike', 'option_type', 'expiry_date', 
                        'price', 'volume', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume']]
        client.insert_ticks(insert_df)
        logger.info("Ingestion complete.")
    except Exception as e:
        logger.error(f"Database error: {e}")
    finally:
        client.disconnect()

@flow(name="Daily Options Data Pipeline", log_prints=True)
def daily_options_pipeline(target_date: str | None = None):
    """
    Main Prefect Flow triggered daily at 14:00.
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    print(f"Starting daily pipeline for {target_date}")
    
    raw_data = fetch_shioaji_data(target_date)
    cleaned_data = clean_and_impute_data(raw_data)
    iv_data = calculate_iv_surface(cleaned_data)
    ingest_to_timescaledb(iv_data)
    
    print("Pipeline execution finished successfully.")

if __name__ == "__main__":
    # To run this daily at 14:00, you would deploy this via Prefect:
    # prefect deployment build daily_pipeline.py:daily_options_pipeline -n "daily_shioaji" --cron "0 14 * * *" -a
    daily_options_pipeline()