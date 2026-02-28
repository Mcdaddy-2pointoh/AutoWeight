# Standard Libs
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Specialised imports
from src.utils.text_operators import format_error_text, format_info_text, format_success_text, format_warn_text
from src.utils.file_operators import load_latest_dataframe, load_yaml, save_dataframe
from src.utils.validators import validate_create_portfolio_universe_config, ConfigValidationError


def create_portfolio_universe(
        config_path: str | Path, 
        save_data: bool = True,
        verbose: bool = False
)  -> Dict[str, pd.DataFrame]:
    """
    Function: Imports the pair level correlation calculated for each ticker and creates a universe of all tickers with low correlation to select
    Args:
        config_path (str | Path): Path to the config file
        save_data (bool): Saves the data at specified path
        verbose (bool): Prints the operation status within the function
    """

    if verbose:
        print(format_info_text("Loading Portfolio Params"))

    # Load the config file
    config = load_yaml(Path(config_path))
    data_path_config = config['config']['data_path']
    corr_universe_params = config['config']['parameters']['portfolio_params']

    # Validate the config params
    try:
        validate_create_portfolio_universe_config(corr_universe_params)

    except ConfigValidationError as e:
        print(format_error_text(f"""Portfolio Parameters validation failed: {str(e)}"""))
        raise RuntimeError(format_error_text("Portfolio Universe Selection Pipeline Failed"))

    except Exception as e:
        print(format_error_text("Failed to load the correlation config parameters"))
        raise RuntimeError(format_error_text("Portfolio Universe Selection Pipeline Failed"))

    if verbose:
        print(format_success_text("Loaded and Validated Portfolio Params"))
        print()

    # Load the non filtered ticker x ticker matrix to calculate global corr correlation plot for adj_close
    corr_martix_df = load_latest_dataframe(
        directory=data_path_config['correlation_matrix']['directory'],
        file_name=data_path_config['correlation_matrix']['file_name'],
        file_format=data_path_config['correlation_matrix']['file_format'],
        suffix="adj_close"
    )

    # Process the correlation matrix
    corr_martix_df.rename(columns={'Unnamed: 0': 'securities'}, inplace=True)
    corr_martix_df.set_index(corr_martix_df['securities'], drop=True, inplace=True)
    corr_martix_df.drop(columns=['securities'], inplace=True)

    # Print the gloabl correlation caluclation has begun
    if verbose:
        print(format_success_text("Loaded the latest correlation Matrix"))
        print()
        print(format_info_text("Calculating the Global Correlation for all securities in correlation matrix"))

    # Process values such that the diagonal elements are considered as NaN, to eliminate self correlation
    np.fill_diagonal(abs_corr_martix_df.values, np.nan)

    # Average Global correlation
    global_corr = abs_corr_martix_df.mean(axis=1)
    global_corr_sorted = global_corr.sort_values(ascending=True)
    anchors = global_corr_sorted
    
    # If the len of global_corr_sorted is less tham the anchor points specifed in the config raise a warning
    if len(global_corr_sorted) < corr_universe_params['correlation_anchor_points']:
        print(format_warn_text(f"The number of correlation_anchor_points is mmore than the legth of global correlation matrix, {len(global_corr_sorted)} anchor points will be used"))

    # Else filter the global correlation
    else:       
        if verbose:
            print(format_info_text(f"Filtering global correlation to loowest globally correlated {corr_universe_params['correlation_anchor_points']} anchor points"))
        anchors = anchors[:corr_universe_params['correlation_anchor_points']]

    # GC
    del abs_corr_martix_df
    del global_corr
    del corr_martix_df

    # Print the gloabl correlation caluclation has begun
    if verbose:
        print(format_success_text("Global Correlation for all securities caluclated"))
        print() 
        print(format_info_text("Finding Optimal Buckets"))

    return anchors