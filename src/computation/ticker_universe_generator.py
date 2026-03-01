# Standard Libs
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

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

    # Calculate the absolute value for the correlation matrix, to eliminate directionality
    abs_corr_martix_df = np.abs(corr_martix_df)

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
    del global_corr
    del corr_martix_df

    # Print the gloabl correlation caluclation has begun
    if verbose:
        print(format_success_text("Global Correlation for all securities caluclated"))
        print() 
        print(format_info_text("Finding Optimal Buckets"))

    # Generate Candidate Portfolios
    candidate_portfolios = greedy_portfolio_select(config_path=config_path, abs_corr_martix_df=abs_corr_martix_df, anchors=anchors, verbose=verbose)

    return anchors, candidate_portfolios

def greedy_portfolio_select(
        config_path,
        abs_corr_martix_df,
        anchors: pd.Series,
        verbose: bool = False
):
    """
    Function: Accepts the anchor security and creates a candidate portfolio around the same, returning the list of equities, corr at addition of each and avg portfolio corr
    Args:
        config_path (str | Path): Path to the config file
        anchors (List[str]): Portfolio initializing securities
        abs_corr_martix_df: Absolute adjusted correlation martrix 
    Returns:
        candidate_portfolios (dict): 
            portfolio_key (List[str]): All securities in the portfolio
            portfolio_corr (List[float]): Evolution of corr of portfolio with addition of corresponding securities
    """

    # Load the config params
    if verbose:
        print(format_info_text("Loading Config Params for Portfolio Selection"))
        print()
    config = load_yaml(Path(config_path))
    config_params = config['config']['parameters']['portfolio_params']
    if verbose:
        print(format_success_text("Loaded Config Params for Portfolio Selection"))
        print()

    # Define an emptiy 
    candidate_portfolios = {}

    # For each anchor create a candidate portfolio
    for anchor in anchors.index:

        if verbose:
            print(format_info_text(f"Creating portfolio for anchor: {anchor}"))
            print()

        # Set the portfolio list
        all_securities_list = list(abs_corr_martix_df.index)
        port_list = [anchor]
        selection_list = [sec for sec in all_securities_list if sec not in port_list]

        # Corr progression list
        corr_prog = [0]

        # Loop thru all equities not in port list to identify the one with the lowest corr
        while(len(selection_list) > 0):

            # Check if len of portfolio == to config len
            # If so break the loop
            if len(port_list) == config_params['max_bucket_size']:     
                if verbose:
                    print(format_success_text(f"Portfolio Created for anchor: {anchor}"))
                    print(format_info_text(f"Termination achieved, max portfolio size hit: {config_params['max_bucket_size']}"))
                    print()
                break

            else:
                # Subset to get the current portfolio columns
                temp_corr_df = abs_corr_martix_df[port_list]

                # Filter to keep rows that aren't in portfolio
                temp_corr_df = pd.DataFrame(temp_corr_df[temp_corr_df.index.isin(selection_list)])

                # Get the max of all corrs across columns
                temp_corr_df['max'] = temp_corr_df.max(axis=1)

                # Get the min value and the min index value
                min_val = temp_corr_df['max'].min()
                min_idx = temp_corr_df['max'].idxmin()

                # If the minimum value is greater than the corr threshold break the loop
                if min_val > config_params['corr_threshold']:
                    if verbose:
                        print(format_success_text(f"Portfolio Created for anchor: {anchor}"))
                        print(format_info_text(f"Termination achieved, exhaustion of low correlation pairs below corr: {config_params['corr_threshold']}"))
                        print(format_info_text(f"Portfolio Size: {len(port_list)}"))
                        print()
                    break

                # Else add the security to portfolio
                # Remove the security from selction list
                else:
                    selection_list.remove(str(min_idx))
                    port_list.append(str(min_idx))
                    corr_prog.append(min_val)

                    if len(selection_list) == 0:
                        if verbose:
                            print(format_success_text(f"Portfolio Created for anchor: {anchor}"))
                            print(format_info_text(f"Termination achieved, no security left for selection"))
                            print(format_info_text(f"Portfolio Size: {len(port_list)}"))
                            print()
                        break                      
        
        # Append the optimal portfolio to candidate
        candidate_portfolios[anchor] = {}
        candidate_portfolios[anchor]['candidate_portfolio'] = port_list
        candidate_portfolios[anchor]['candidate_corr_prog'] = corr_prog

    return candidate_portfolios