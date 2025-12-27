# Standard Libs
import pandas as pd
import numpy as np
from pathlib import Path

# Specialised imports
from src.utils.text_operators import format_error_text, format_info_text, format_success_text
from src.utils.file_operators import load_latest_dataframe, load_yaml
from src.utils.validators import validate_correlation_config

def calculate_correlation(
        config_path: str | Path, 
        save_data: bool = True
    ) -> pd.DataFrame:

    """
    Function: Computes the correlation for all metrics listed in the config
    Args:
        config_path (str | Path): Path to the config file
        save_data (bool): Saves the data at specified path
    """

    # Load the config file
    config = load_yaml(Path(config_path))
    data_config = config['config']['data']
    data_path_config = config['config']['data_path']
    correlation_parameters_config = config['config']['parameters']['correlation']

    # Validate the config params
    try:
        validate_correlation_config(correlation_parameters_config)

    except ValueError as e:
        print(format_error_text(f"""Correlation parameters validation failed: {str(e)}"""))

    except Exception as e:
        print(format_error_text("Failed to load the correlation config parameters"))

    # Convert the metrics to a list
    if isinstance(correlation_parameters_config['metrics'], str):
        correlation_parameters_config['metrics'] = [correlation_parameters_config['metrics']]

    # Loop through all the metrics
    # Choose the latest df from the data at path 
    # Calculate the correlation of columns
    for metric in correlation_parameters_config['metrics']:

        # Log the info
        print(format_info_text(f"Calculating Corrlation for {metric} metric of data"))

        # Metric to path mapper
        metric_path_mapper = {
            "open": "ticker_data_open",
            "high": "ticker_data_high",
            "low": "ticker_data_low",
            "close": "ticker_data_close",
            "volume": "ticker_data_volume"
        }

        metric_key = metric_path_mapper[metric]

        # Load the latest dataframe
        temp_metric_df = load_latest_dataframe(
            directory=data_path_config[metric_key]['directory'],
            file_name=data_path_config[metric_key]['file_name'],
            file_format=data_path_config[metric_key]['file_format'],
        )

        # 


