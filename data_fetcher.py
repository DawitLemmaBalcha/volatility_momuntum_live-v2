import os
import pandas as pd
from pybit.unified_trading import HTTP
import time
import config
import logging

# It's good practice to have a logger for the data fetching process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _fetch_from_bybit(session, category: str, symbol: str, timeframe: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
    """
    Internal function to fetches historical OHLCV data from Bybit for a specific timeframe.
    Handles pagination by working backwards from the end date to retrieve all data.
    """
    all_ohlcv = []
    current_end_ms = end_time_ms
    logging.info(f"Fetching from Bybit: {symbol} ({timeframe}) from {pd.to_datetime(start_time_ms, unit='ms')} to {pd.to_datetime(current_end_ms, unit='ms')}")
    
    while True:
        try:
            response = session.get_kline(
                category=category, symbol=symbol, interval=timeframe,
                end=current_end_ms, limit=1000
            )
            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                oldest_timestamp = int(data[-1][0])
                valid_data = [d for d in data if int(d[0]) >= start_time_ms]
                all_ohlcv.extend(valid_data)
                logging.info(f"  ...fetched {len(valid_data)} candles back to {pd.to_datetime(oldest_timestamp, unit='ms')}")
                
                if oldest_timestamp < start_time_ms or len(valid_data) == 0:
                    break
                
                current_end_ms = oldest_timestamp - 1
                time.sleep(0.2)
            else:
                logging.warning(f"No more data received or API error: {response.get('retMsg', 'Unknown Error')}")
                break
        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}")
            time.sleep(1)
            continue
            
    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.drop_duplicates(subset='timestamp').sort_values(by='timestamp', ascending=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def _get_data_for_timeframe(timeframe: str, req_start_dt: pd.Timestamp, req_end_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Handles the intelligent fetching and caching for a single timeframe.
    """
    symbol_safe = config.SYMBOL.replace('/', '')
    master_cache_path = f"data/{symbol_safe}_{timeframe}_master.parquet"
    
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Case 1: Master cache file exists
    if os.path.exists(master_cache_path):
        logging.info(f"Master cache found for {timeframe}: {master_cache_path}")
        df_master = pd.read_parquet(master_cache_path)
        cache_start_dt = df_master['timestamp'].min()
        cache_end_dt = df_master['timestamp'].max()

        # If requested range is fully within the cache, just slice and return
        if req_start_dt >= cache_start_dt and req_end_dt <= cache_end_dt:
            logging.info(f"Requested range is within cache. Slicing data.")
            return df_master[(df_master['timestamp'] >= req_start_dt) & (df_master['timestamp'] <= req_end_dt)].copy()
        
        # If requested range is partially outside, fetch only what's missing
        else:
            logging.info("Cache found, but requested range is partially outside. Fetching missing data.")
            df_to_append = pd.DataFrame()
            
            # Fetch older data if needed
            if req_start_dt < cache_start_dt:
                df_to_append = pd.concat([df_to_append, _fetch_from_bybit(HTTP(), "linear", symbol_safe, timeframe, int(req_start_dt.timestamp() * 1000), int((cache_start_dt).timestamp() * 1000))])
            
            # Fetch newer data if needed
            if req_end_dt > cache_end_dt:
                df_to_append = pd.concat([df_to_append, _fetch_from_bybit(HTTP(), "linear", symbol_safe, timeframe, int((cache_end_dt).timestamp() * 1000), int(req_end_dt.timestamp() * 1000))])

            if not df_to_append.empty:
                df_master = pd.concat([df_master, df_to_append]).drop_duplicates(subset='timestamp').sort_values(by='timestamp', ascending=True)
                df_master.to_parquet(master_cache_path)
                logging.info(f"Master cache for {timeframe} updated and saved.")
            
            return df_master[(df_master['timestamp'] >= req_start_dt) & (df_master['timestamp'] <= req_end_dt)].copy()
    
    # Case 2: No master cache file exists, download for the first time
    else:
        logging.info(f"No master cache found for {timeframe}. Performing initial download.")
        df_new = _fetch_from_bybit(HTTP(), "linear", symbol_safe, timeframe, int(req_start_dt.timestamp() * 1000), int(req_end_dt.timestamp() * 1000))
        if not df_new.empty:
            df_new.to_parquet(master_cache_path)
            logging.info(f"Initial master cache for {timeframe} created and saved.")
        return df_new

def fetch_all_dataframes():
    """
    Main function to fetch only 1-minute data using the intelligent cache (30m and 4h are derived dynamically).
    """
    # Get requested date range from config
    req_start_dt = pd.to_datetime(config.START_DATE)
    req_end_dt = pd.to_datetime(config.END_DATE)
    logging.info(f"Data request for {config.SYMBOL} from {req_start_dt} to {req_end_dt}")

    # Fetch only 1m data; return empty for 30m and 4h
    df_1m = _get_data_for_timeframe('1', req_start_dt, req_end_dt)
    df_30m = pd.DataFrame()  # Empty, as derived from 1m
    df_4h = pd.DataFrame()   # Empty, unused
    
    if df_1m.empty:
        raise ValueError("Could not fetch 1m data.")
        
    logging.info(f"Successfully loaded {len(df_1m)} 1m candles (30m and 4h derived dynamically).")
    return df_30m, df_1m, df_4h