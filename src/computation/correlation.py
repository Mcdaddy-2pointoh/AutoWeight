# Standard Libs
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Specialised imports
from src.utils.text_operators import format_error_text, format_info_text, format_success_text, format_warn_text
from src.utils.file_operators import load_latest_dataframe, load_yaml, save_dataframe
from src.utils.validators import validate_pairwise_correlation_config, ConfigValidationError

def pct_change_calculator(df: pd.DataFrame, drop_na: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Function: Converts absolute prices in a dataframe to a pct change
    Args:
        df (pd.DataFrame): The dataframe containing the prices as values, commodities as columns and indexed on the date
        drop_na (bool): Default False, drops NA from the pct column if set to true
    Returns:
        pd.DataFrame
    """
    
    if drop_na:
        return df.pct_change().dropna()

    else:
        return df.pct_change()


def calculate_pairwise_corrrelation(
        config_path: str | Path, 
        save_data: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:

    """
    Function: Computes the pair level correlation for all TICEKRS listed in the config
    Args:
        config_path (str | Path): Path to the config file
        save_data (bool): Saves the data at specified path
    """

    # Load the config file
    config = load_yaml(Path(config_path))
    data_path_config = config['config']['data_path']
    correlation_parameters_config = config['config']['parameters']['correlation']

    # Validate the config params
    try:
        validate_pairwise_correlation_config(correlation_parameters_config)

    except ConfigValidationError as e:
        print(format_error_text(f"""Correlation parameters validation failed: {str(e)}"""))
        raise RuntimeError(format_error_text("Correlation Pipeline Failed"))

    except Exception as e:
        print(format_error_text("Failed to load the correlation config parameters"))
        raise RuntimeError(format_error_text("Correlation Pipeline Failed"))


    # Convert the metrics to a list
    if isinstance(correlation_parameters_config['metrics'], str):
        correlation_parameters_config['metrics'] = [correlation_parameters_config['metrics']]

    correlation_results = {}

    # Loop through all the metrics
    # Choose the latest df from the data at path 
    # Calculate the correlation of columns
    for metric in correlation_parameters_config['metrics']:

        # Log the info
        if verbose:
            print(format_info_text(f"   Starting Correlation Pipeline for {metric} metric of data"))

        # Load the latest dataframe
        temp_metric_df = load_latest_dataframe(
            directory=data_path_config['ticker_data']['directory'],
            file_name=data_path_config['ticker_data']['file_name'],
            file_format=data_path_config['ticker_data']['file_format'],
            suffix=metric
        )

        # Set date to datetime object
        # Set date as the index
        
        temp_metric_df["Date"] = pd.to_datetime(temp_metric_df["Date"])
        temp_metric_df.set_index(keys="Date", inplace=True)

        # Convert prices to pct changes for better correlation calculation
        if verbose:
            print(format_info_text(f"   Calculating PCT Change for {metric} metric of data"))

        pct_changed_df = pct_change_calculator(df=temp_metric_df, drop_na=False)

        # Calculate correlation
        if verbose:
            print(format_info_text(f"   Calculating Correlation for PCT Change for {metric} metric of data"))

        corr_matrix = pct_changed_df.corr(method=correlation_parameters_config['method'])

        # Calculate the upper triangular matrix for getting unique values only for corr
        pairs_df = (
            corr_matrix
            .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs_df.columns = ["ticker_1", "ticker_2", "correlation"]

        # Validate the Count of observations available for a pair
        valid_pairs = []
        for t1, t2 in pairs_df[["ticker_1", "ticker_2"]].values:
            n_obs = pct_changed_df[[t1, t2]].dropna().shape[0]
            if n_obs >= correlation_parameters_config['min_observations']:
                valid_pairs.append((t1, t2))

        if verbose:
            print(format_info_text(f"   Adjusting Correlation based on optimization strategy for {metric} metric of data"))


        # Adjust correlation for the optimization strategy defined & identify pairs
        pairs_df["adjusted_corr"] = pairs_df["correlation"].abs()
        pairs_df.sort_values("adjusted_corr", inplace=True)
        pairs_df.reset_index(drop=True, inplace=True)

        # Reset index for filtered pairs
        pairs_df = pairs_df[['ticker_1', 'ticker_2', 'correlation', 'adjusted_corr']]
        pairs_df.reset_index(drop=True, inplace=True)

        # Calculate the absolute correlation
        pairs_df['abs_corr'] = np.abs(pairs_df['correlation'])

        # Save the intermediate files
        if save_data:

            # Save the pct_change_matrix
            save_dataframe(
                df=pct_changed_df,
                directory=data_path_config['pct_change_matrix']['directory'],
                file_name=data_path_config['pct_change_matrix']['file_name'],
                file_format=data_path_config['pct_change_matrix']['file_format'],
                suffix=metric,
                versioned=True,
                save_index=True
            )

            # Save the corr_matrix
            save_dataframe(
                df=corr_matrix,
                directory=data_path_config['correlation_matrix']['directory'],
                file_name=data_path_config['correlation_matrix']['file_name'],
                file_format=data_path_config['correlation_matrix']['file_format'],
                suffix=metric,
                versioned=True,
                save_index=True
            )

            # Save all pairs and correlations
            save_dataframe(
                df=pairs_df,
                directory=data_path_config['adjusted_pairs_matrix']['directory'],
                file_name=data_path_config['adjusted_pairs_matrix']['file_name'],
                file_format=data_path_config['adjusted_pairs_matrix']['file_format'],
                suffix=metric,
                versioned=True,
                save_index=False
            )

            if verbose:
                print(format_success_text(f"   Correlation Pipeline for {metric} metric of dataframes, are saved."))

        
        correlation_results[metric] = {
            "pct_changed_df" : pct_changed_df,
            "corr_matrix" : corr_matrix,
            "pairs_df" : pairs_df,
        }

        if verbose:
            print(format_success_text(f"   Correlation Pipeline for {metric} metric of data finished successfully"))
            print("")


    print(format_success_text(f"Correlation Pipeline complete!"))
    return correlation_results