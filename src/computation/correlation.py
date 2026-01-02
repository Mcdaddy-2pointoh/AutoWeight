# Standard Libs
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from typing import Dict

# Specialised imports
from src.utils.text_operators import format_error_text, format_info_text, format_success_text, format_warn_text
from src.utils.file_operators import load_latest_dataframe, load_yaml, save_dataframe
from src.utils.validators import validate_correlation_config

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


def calculate_correlation(
        config_path: str | Path, 
        save_data: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:

    """
    Function: Computes the correlation for all metrics listed in the config
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
        validate_correlation_config(correlation_parameters_config)

    except ValueError as e:
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

        pct_changed_df = pct_change_calculator(df=temp_metric_df, drop_na=True)

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
        if correlation_parameters_config['optimization_strategy'].lower() == "low":
            pairs_df["adjusted_corr"] = pairs_df["correlation"].abs()
            pairs_df.sort_values("adjusted_corr", inplace=True)
            pairs_df.reset_index(drop=True, inplace=True)


        elif correlation_parameters_config['optimization_strategy'].lower() == "negative":
            pairs_df['adjusted_corr'] = np.where(pairs_df["correlation"] < 0, 
                                                 pairs_df["correlation"].abs(),
                                                 0)
            pairs_df.sort_values("adjusted_corr", ascending=False, inplace=True)
            pairs_df.reset_index(drop=True, inplace=True)
                  
        else:
            print(format_error_text(f"  Optimization strategy {correlation_parameters_config['optimization_strategy'].lower()} is invalid. Choose strategy `negative` or `low`"))
            raise RuntimeError(format_error_text("Correlation Pipeline Failed"))


        # If filter flag is true
        if correlation_parameters_config['filter']['filter_n_pairs'] or correlation_parameters_config['filter']['filter_inverse_threshold']:
            
            # If both flags are true 
            if correlation_parameters_config['filter']['filter_n_pairs'] and correlation_parameters_config['filter']['filter_inverse_threshold']:

                # Inverse threshold works only when strategy is negative
                if correlation_parameters_config['optimization_strategy'].lower() == "negative":
                    
                    # Filter the pairs based on inverse threshold
                    filtered_pairs_df = pairs_df[pairs_df['adjusted_corr'] > correlation_parameters_config['filter']['inverse_threshold']]
                    filtered_pairs_df.sort_values("adjusted_corr", ascending=False, inplace=True) 

                    # If the result is an empty matrix
                    if filtered_pairs_df.empty:
                        print(format_warn_text(f"  Filtering strategy due to `inverse_threshold` yeilds an empty dataframe. Reduce the threshold"))

                    # If rows less than the top_n_pairs value
                    elif filtered_pairs_df.shape[0] < correlation_parameters_config['filter']['top_n_pairs']:
                        print(format_warn_text(f"  Number of filtered pairs are less than the `top_n_pairs` config set at {correlation_parameters_config['filter']['top_n_pairs']}. Saving all pairs"))

                    else:
                        filtered_pairs_df = filtered_pairs_df[:correlation_parameters_config['filter']['top_n_pairs']]
                
                # Else raise an error
                else:
                    print(format_error_text(f"  Optimization strategy {correlation_parameters_config['optimization_strategy'].lower()} cannot be used with filter `filter_inverse_threshold`"))
                    raise RuntimeError(format_error_text("Correlation Pipeline Failed"))


            # If only n_pairs are true
            elif correlation_parameters_config['filter']['filter_n_pairs']:

                filtered_pairs_df = pairs_df
                filtered_pairs_df.sort_values("adjusted_corr", ascending=True, inplace=True) 
                
                # If rows less than the top_n_pairs value
                if filtered_pairs_df.shape[0] < correlation_parameters_config['filter']['top_n_pairs']:
                    print(format_warn_text(f"  Number of filtered pairs are less than the `top_n_pairs` config set at {correlation_parameters_config['filter']['top_n_pairs']}. Saving all pairs"))

                else:
                    filtered_pairs_df = filtered_pairs_df[:correlation_parameters_config['filter']['top_n_pairs']]


            elif correlation_parameters_config['filter']['filter_inverse_threshold']:

                # Filter the pairs based on inverse threshold
                filtered_pairs_df = pairs_df[pairs_df['adjusted_corr'] > np.abs(correlation_parameters_config['filter']['inverse_threshold'])]
                filtered_pairs_df.sort_values("adjusted_corr", ascending=False, inplace=True) 

                # If the result is an empty matrix
                if filtered_pairs_df.empty:
                        print(format_warn_text(f"  Filtering strategy due to `inverse_threshold` yeilds an empty dataframe. Reduce the threshold"))

        else:
            filtered_pairs_df = pairs_df

        # Reset index for filtered pairs
        pairs_df = pairs_df[['ticker_1', 'ticker_2', 'correlation', 'adjusted_corr']]
        pairs_df.reset_index(drop=True, inplace=True)

        # Reset index for filtered pairs
        filtered_pairs_df = filtered_pairs_df[['ticker_1', 'ticker_2', 'correlation', 'adjusted_corr']]
        filtered_pairs_df.reset_index(drop=True, inplace=True)

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

            # Save filtered pairs
            save_dataframe(
                df=filtered_pairs_df,
                directory=data_path_config['filtered_pairs_matrix']['directory'],
                file_name=data_path_config['filtered_pairs_matrix']['file_name'],
                file_format=data_path_config['filtered_pairs_matrix']['file_format'],
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
            "filtered_pairs_df" : filtered_pairs_df
        }

        if verbose:
            print(format_success_text(f"   Correlation Pipeline for {metric} metric of data finished successfully"))
            print("")


    print(format_success_text(f"Correlation Pipeline complete!"))
    return correlation_results
    
