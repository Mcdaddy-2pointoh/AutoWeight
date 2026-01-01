import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union
from pathlib import Path
from src.utils.file_operators import load_yaml, save_dataframe
from datetime import datetime, timedelta
from src.utils.text_operators import format_error_text, format_info_text, format_success_text

def fetch_data(
        config_path: str | Path, 
        save_data: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:
    
    """
    Function: Accepts and loads the config file from the specified path. Downloads the ticker level data and produces a version controlled dataframe of all the tickers metrics (OHLCV) 
    Args:
        config_path (str | Path): Path to the config file
        save_data (bool): Saves the data at specified path
    """

    # Load the config file
    conf = load_yaml(Path(config_path))

    # Load the config values
    data_config = conf['config']['data']
    data_path_config = conf['config']['data_path']

    # Set the start and the end dates
    end_date = datetime.today()
    start_date = end_date - timedelta(days = data_config['window_in_days'])

    # Create a ohlcv_df 
    ohlcv_data = {
        "ticker_list" : data_config['tickers'],
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
        "adj_close" : None
    }

    # For each ticker add the OHLCV value for a daily level
    for ticker in ohlcv_data['ticker_list']:

        if verbose:
            print(format_info_text(f"Downloading data for ticker: {ticker}"))

        # Load the OHLCV data
        try:
            temp_df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            if verbose:
                print(format_success_text(f"Successfully downloaded data for: {ticker}"))

        except:
            print(format_error_text(f"Failed to load data for: {ticker}"))

        if verbose:
            print(format_info_text(f"Processing data for ticker: {ticker}"))

        # Split the data frame into series
        open_temp_series = temp_df['Open'].copy(deep=True)
        close_temp_series = temp_df['Close'].copy(deep=True)
        high_temp_series = temp_df['High'].copy(deep=True)
        low_temp_series = temp_df['Low'].copy(deep=True)
        volume_temp_series = temp_df['Volume'].copy(deep=True)
        adj_close_temp_series = temp_df['Adj Close'].copy(deep=True)

        # Convert the series to a dataframe 
        open_temp_df = pd.DataFrame(open_temp_series).rename(columns={'Open': ticker})
        close_temp_df = pd.DataFrame(close_temp_series).rename(columns={'Close': ticker})
        high_temp_df = pd.DataFrame(high_temp_series).rename(columns={'High': ticker})
        low_temp_df = pd.DataFrame(low_temp_series).rename(columns={'Low': ticker})
        volume_temp_df = pd.DataFrame(volume_temp_series).rename(columns={'Volume': ticker})
        adj_close_temp_df = pd.DataFrame(adj_close_temp_series).rename(columns={'Adj Close': ticker})

        # Check if open data frame is None
        # If none use the open_temp_df as a main df
        # Else join using index 
        if ohlcv_data['open'] is None:
            ohlcv_data['open'] = open_temp_df

        else:
            old_df = ohlcv_data['open'].copy(deep=True)
            merged_df = pd.merge(
                left=old_df,
                right=open_temp_df,
                right_index=True,
                left_index=True
            )
            ohlcv_data['open'] = merged_df

            del old_df
            del merged_df

        # Check if high data frame is None
        # If none use the high_temp_df as a main df
        # Else join using index 
        if ohlcv_data['high'] is None:
            ohlcv_data['high'] = high_temp_df

        else:
            old_df = ohlcv_data['high'].copy(deep=True)
            merged_df = pd.merge(
                left=old_df,
                right=high_temp_df,
                right_index=True,
                left_index=True
            )
            ohlcv_data['high'] = merged_df

            del old_df
            del merged_df

        # Check if low data frame is None
        # If none use the low_temp_df as a main df
        # Else join using index 
        if ohlcv_data['low'] is None:
            ohlcv_data['low'] = low_temp_df

        else:
            old_df = ohlcv_data['low'].copy(deep=True)
            merged_df = pd.merge(
                left=old_df,
                right=low_temp_df,
                right_index=True,
                left_index=True
            )
            ohlcv_data['low'] = merged_df

            del old_df
            del merged_df   

        # Check if close data frame is None
        # If none use the close_temp_df as a main df
        # Else join using index 
        if ohlcv_data['close'] is None:
            ohlcv_data['close'] = close_temp_df

        else:
            old_df = ohlcv_data['close'].copy(deep=True)
            merged_df = pd.merge(
                left=old_df,
                right=close_temp_df,
                right_index=True,
                left_index=True
            )
            ohlcv_data['close'] = merged_df

            del old_df
            del merged_df 

        # Check if volume data frame is None
        # If none use the volume_temp_df as a main df
        # Else join using index 
        if ohlcv_data['volume'] is None:
            ohlcv_data['volume'] = volume_temp_df

        else:
            old_df = ohlcv_data['volume'].copy(deep=True)
            merged_df = pd.merge(
                left=old_df,
                right=volume_temp_df,
                right_index=True,
                left_index=True
            )
            ohlcv_data['volume'] = merged_df

            del old_df
            del merged_df


        # Check if adj_close data frame is None
        # If none use the adj_close_temp_df as a main df
        # Else join using index 
        if ohlcv_data['adj_close'] is None:
            ohlcv_data['adj_close'] = adj_close_temp_df

        else:
            old_df = ohlcv_data['adj_close'].copy(deep=True)
            merged_df = pd.merge(
                left=old_df,
                right=adj_close_temp_df,
                right_index=True,
                left_index=True
            )
            ohlcv_data['adj_close'] = merged_df

            del old_df
            del merged_df

        if verbose:
            print(format_success_text(f"Successfully processed data for ticker: {ticker}"))
            print("")

    if save_data:

        # Log the process
        if verbose:
            print(format_info_text(f"Saving data for all tickers:"))
        for idx, ticker in enumerate(ohlcv_data['ticker_list']):

            if verbose:
                print(format_info_text(f"   {idx+1}. {ticker}"))

        # Convert all the keys to a dataframe
        ohlcv_data['ticker_list'] = pd.DataFrame(ohlcv_data['ticker_list'], columns=["ticker"])

        # Save the data frames
        for key in ohlcv_data.keys():

            # Load the key dataframe
            # Load the key config 
            if key == "ticker_list":
                file_key = "ticker_list"

            else:
                file_key = "ticker_data"

            df = ohlcv_data[key]

            # Save the dateframe
            save_dataframe(
                df,
                directory=data_path_config[file_key]['directory'],
                file_name=data_path_config[file_key]['file_name'],
                file_format=data_path_config[file_key]['file_format'],
                versioned=data_path_config[file_key]['versioned'],
                suffix=key,
                save_index=True
            )

        if verbose:
            print(format_success_text(f"Successfully saved data for tickers"))
        print(format_success_text(f"fetch_data pipeline run completed"))
        return ohlcv_data

    else:
        print(format_success_text(f"fetch_data pipeline run completed"))
        return ohlcv_data
