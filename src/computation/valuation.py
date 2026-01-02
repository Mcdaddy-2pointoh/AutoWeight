# Standard libraries for imports
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path

# Import internal libs
from src.utils.file_operators import load_latest_dataframe, load_yaml, save_dataframe
from src.utils.text_operators import format_error_text, format_info_text, format_success_text, format_warn_text
from src.utils.validators import validate_valuation_config

def calculate_log_column(df: pd.DataFrame, column_names: str | list) -> pd.DataFrame:
    """
    Function: Using the `column_name` column of the DataFrame df creates a log price value for the same
    """
    
    if isinstance(column_names, str):
        df[f'log_{column_names}'] = np.log(df[column_names])

    elif isinstance(column_names, list):
        for column in column_names:
            df[f'log_{column}'] = np.log(df[column])

    return df

def calculate_score(df: pd.DataFrame, 
                    column_name: str, 
                    window: int, 
                    clip: float | None = 0,
                    apply_bounding: bool = False):
    """
    Function: Calculates the valuation score based on the valuation methodology
    Args:
        df (pd.DataFrame): Price DF with all the calculated metric column
        window (int): Rolling mean window
        column_name (str): Column name to apply score on
        clip (float | None): Cipping values for score
        apply_bounding (bool): Bounds the valuation score between -1 and 1 using tanh
    """


    rolling_mean = df[column_name].rolling(window).mean()
    rolling_std = df[column_name].rolling(window).std()

    z_score = (df[column_name] - rolling_mean) / rolling_std

    # Undervaluation logic
    if column_name in {"price", "log_price"}:
        score = (-z_score)

    else:  # log_return
        # Underperformance = negative cumulative return
        cumulative = df[column_name].rolling(window).sum()
        score = (-cumulative / rolling_std)

    df[f'bounded_valuation_score_{column_name}'] = np.tanh(score)

    if clip is not None:
        df[f'valuation_score_{column_name}'] = score.clip(lower=clip)

    else: 
        df[f'valuation_score_{column_name}'] = score

    return df 


def raw_valuation_score_computation(config_path: str | Path, 
                            save_data: bool = True,
                            verbose: bool = False
                            ) -> Dict[str, pd.DataFrame]:
    """
    Function: Computes the valuation of an equity relative to its prior prices
    Args:
        config_path (str | Path): Path to the config file
        save_data (bool): Saves the data at specified path
    """

    if verbose:
        print(format_info_text("Starting Valuation Pipeline."))

    # Load config values
    config = load_yaml(Path(config_path))
    data_path_config = config['config']['data_path']
    valuation_parameters_config = config['config']['parameters']['valuation']
    
    # If clip is a value in config
    clip = None
    if 'clip' in valuation_parameters_config.keys():
        clip = valuation_parameters_config['clip']

    # Validate valuation params
    try:
        validate_valuation_config(valuation_parameters_config)

    except ValueError as e:
        print(format_error_text(f"""Validation parameters validation failed: {str(e)}"""))


    # Load the latest data frame
    if valuation_parameters_config['metric'] != "adj_close":
        print(format_warn_text(f"Using price metric {valuation_parameters_config['metric']}, the use OHLC prices for long-term valuation is discouraged"))
        if verbose:
            print(format_info_text("    Open: distorted by overnight gaps and news; not a consensus price."))
            print(format_info_text("    High: intraday extreme; inflates volatility."))
            print(format_info_text("    Low: intraday extreme; creates false undervaluation signals."))
            print(format_info_text("    Close: distorted by corporate actions (splits, dividends)."))

        print(format_warn_text("Adjusted Close is recommended as it reflects true adjusted economic returns."))

    else:
        if verbose:
            print(format_info_text(f"Using price metric {valuation_parameters_config['metric']}"))

    # Load the metric df
    metric_df = load_latest_dataframe(
        directory=data_path_config['ticker_data']['directory'],
        file_name=data_path_config['ticker_data']['file_name'],
        file_format=data_path_config['ticker_data']['file_format'],
        suffix=valuation_parameters_config['metric']
    )

    # Set date to datetime object
    # Set date as the index
    
    metric_df["Date"] = pd.to_datetime(metric_df["Date"])
    metric_df.set_index(keys="Date", inplace=True)

    # Validate the dataframe is bigger in length than the window in days
    if len(metric_df) < valuation_parameters_config['window_in_days']:
        print(format_error_text("window_in_days parameter in [parameters -> valuation] cannot be greater than the price data frames length."))
        print(format_error_text(f"   Got window_in_days as: {valuation_parameters_config['window_in_days']}"))
        print(format_error_text(f"   Got price_df with length: {len(metric_df)}"))
        raise RuntimeError(format_error_text("Window provided for aggregation greater than the dataframe"))
    
    # Rename Column name with suffix price
    metric_df.rename(
        columns={column_nm : f'price_{column_nm}' for column_nm in metric_df.columns},
        inplace=True
        )
    
    price_columns = [column for column in metric_df.columns if str(column).startswith("price_")]

    # Compute the log prices and returns if need be and collect all columns to apply the z-score on
    method = valuation_parameters_config['method']
    z_score_columns = []

    # Calculate log price if method is 'all' or 'log_price_z_score
    if method in ['log_price_z_score', 'all']: 
        # Check and drop nulls
        if metric_df.shape != metric_df.dropna().shape:
            print(format_warn_text("NULLs found in price_df, dropping NULLs before calculating `log_price`"))

            if verbose:
                print(format_info_text(f"Shape of the price_df was {metric_df.shape}, before dropping NULLS"))

            metric_df = metric_df.dropna()

            if verbose:
                print(format_info_text(f"Shape of the price_df was {metric_df.shape}, after dropping NULLS"))

        metric_df = calculate_log_column(df= metric_df, column_names=price_columns)
        
        # Only add log_price_ prefixed columns into the list 
        for column in metric_df.columns:

            if str(column).startswith("log_price_"):
                z_score_columns.append(column)
        
        if verbose:
            print(format_info_text(f"Calculated log_price from the {valuation_parameters_config['metric']} price metric"))

    # Calculate returns then the log for returns
    if method in ['log_return_z_score', 'all']:

        # Check and drop nulls
        if metric_df.shape != metric_df.dropna().shape:
            print(format_warn_text("NULLs found in price_df, dropping NULLs before calculating `returns` & `log_returns`"))

            if verbose:
                print(format_info_text(f"Shape of the price_df was {metric_df.shape}, before dropping NULLS"))

            metric_df = metric_df.dropna()

            if verbose:
                print(format_info_text(f"Shape of the price_df was {metric_df.shape}, after dropping NULLS"))


        # Calculate returns
        returns_columns = []
        for column in price_columns:
            ticker_name = str(column).replace("price_", "")
            metric_df[f'returns_{ticker_name}'] = metric_df[column] / metric_df[column].shift(1)    
            metric_df.dropna(inplace=True)
            returns_columns.append(f'returns_{ticker_name}')

        if verbose:
            print(format_info_text(f"Caculated 'returns' from the {valuation_parameters_config['metric']} price metric"))

        # Calculating log returns     
        metric_df = calculate_log_column(metric_df, column_names=returns_columns)
        
        # Add columns prefixed with log_returns_ to the z_score column
        for column in metric_df.columns:
            if str(column).startswith("log_returns_"):
                z_score_columns.append(column)

        if verbose:
            print(format_info_text(f"Caculated 'log_returns' from the 'returns' metric"))


    # Rename the metric column to price for uniformity
    if method in ['price_z_score', 'all']:

        # Check and drop nulls
        if metric_df.shape != metric_df.dropna().shape:
            print(format_warn_text("NULLs found in price_df, dropping NULLs"))

            if verbose:
                print(format_info_text(f"Shape of the price_df was {metric_df.shape}, before dropping NULLS"))

            metric_df = metric_df.dropna()

            if verbose:
                print(format_info_text(f"Shape of the price_df was {metric_df.shape}, after dropping NULLS"))

        # Add columns prefixed with price_ to the z_score column
        for column in metric_df.columns:
            if str(column).startswith("price_"):
                z_score_columns.append(column)

        if verbose:
            print(format_info_text(f"Refactored 'price' from the {valuation_parameters_config['metric']} price metric"))

    if verbose:
        print(format_info_text(f"Computing Valuation scores:"))
        

    # Aggregate the scores
    all_score_df = None
    all_price_columns = []
    all_log_price_columns = []
    all_log_returns_columns = []
    apply_bounding_flag = valuation_parameters_config['apply_bounding']
    for column in list(set(z_score_columns)):

        if verbose:
            print(format_info_text(f"    Computing valuation score for column: {column}"))

        # Column tracker
        if str(column).startswith("price_"):
            all_price_columns.append(column)

        elif str(column).startswith("log_price_"):
            all_log_price_columns.append(column)

        elif str(column).startswith("log_returns_"):
            all_log_returns_columns.append(column)

        # Calculate score
        temp_score_df = calculate_score(df=metric_df, 
                                        column_name=column, 
                                        window=valuation_parameters_config['window_in_days'], 
                                        clip=clip, 
                                        apply_bounding=apply_bounding_flag)

        # If the final score df is none replace with the temp df
        if all_score_df is None:
            all_score_df = temp_score_df

        # Perform a full outer join
        else:

            # Subset the dataframe
            if not apply_bounding_flag:
                temp_score_df = temp_score_df[[f'valuation_score_{column}']]

            else:
                temp_score_df = temp_score_df[[f'valuation_score_{column}', f'bounded_valuation_score_{column}']]

            if verbose:
                print(format_info_text(f"    Merging valuation score for column: {column}"))
                print(format_info_text(f"       Initial DF shape: {all_score_df.shape}"))

            if f'valuation_score_{column}' not in all_score_df.columns:
                all_score_df = pd.merge(
                    left=all_score_df,
                    right=temp_score_df,
                    right_index=True,
                    left_index=True,
                    how="outer"
                )

            if verbose:
                print(format_info_text(f"       Final DF shape: {all_score_df.shape}"))
        
        if verbose:
            print()

    if verbose:
        print(format_success_text("Valuation Scores computed and merged into a single DataFrame"))

    # Calculate the latest scores
    all_score_df.sort_index(inplace=True)
    latest_score_df = all_score_df.iloc[[-1]].copy(deep=True)

    # Remove nulls
    all_score_df.dropna(inplace=True)
    latest_score_df.dropna(inplace=True)

    # Reset the required columns
    if not apply_bounding_flag:
        all_price_columns = [f'valuation_score_{column}' for column in all_price_columns]
        all_log_price_columns = [f'valuation_score_{column}' for column in all_log_price_columns]
        all_log_returns_columns = [f'valuation_score_{column}' for column in all_log_returns_columns]

    else:
        all_price_valuation_columns = [f'valuation_score_{column}' for column in all_price_columns] 
        all_price_bounded_valuation_columns = [f'bounded_valuation_score_{column}' for column in all_price_columns] 
        all_price_columns = all_price_valuation_columns + all_price_bounded_valuation_columns

        all_log_price_valuation_columns = [f'valuation_score_{column}' for column in all_log_price_columns]
        all_log_price_bounded_valuation_columns = [f'bounded_valuation_score_{column}' for column in all_log_price_columns] 
        all_log_price_columns = all_log_price_valuation_columns + all_log_price_bounded_valuation_columns

        all_log_returns_valuation_columns = [f'valuation_score_{column}' for column in all_log_returns_columns]
        all_log_returns_bounded_valuation_columns = [f'bounded_valuation_score_{column}' for column in all_log_returns_columns] 
        all_log_returns_columns = all_log_returns_valuation_columns + all_log_returns_bounded_valuation_columns

    # Save the data created
    if save_data:

        # Save Validation input matrix
        if verbose:
            print(format_info_text("Saving 'valuation_input_matrix' DataFrames"))

        try:
            save_dataframe(
                df=metric_df,
                directory=data_path_config['valuation_input_matrix']['directory'],
                file_name=data_path_config['valuation_input_matrix']['file_name'],
                file_format=data_path_config['valuation_input_matrix']['file_format'],
                save_index=True,
                versioned=True
            )

        except Exception as e:
            print(format_error_text(f"""Failed to save 'valuation_input_matrix': {str(e)}"""))
            raise RuntimeError(format_error_text("Dataframe Save Failed"))


        if verbose:
            print(format_success_text("Saved 'valuation_input_matrix' DataFrames"))

        # Save the overall scores matrix 
        # All scores
        if verbose:
            print(format_info_text("Saving 'valuation_scores_full_matrix' DataFrames"))

        try:
            save_dataframe(
                df=all_score_df,
                directory=data_path_config['valuation_scores_full_matrix']['directory'],
                file_name=data_path_config['valuation_scores_full_matrix']['file_name'],
                file_format=data_path_config['valuation_scores_full_matrix']['file_format'],
                save_index=True,
                versioned=True,
                suffix="all_scores"
            )

        except Exception as e:
            print(format_error_text(f"""Failed to save 'valuation_scores_full_matrix': {str(e)}"""))
            raise RuntimeError(format_error_text("Dataframe Save Failed"))


        if verbose:
            print(format_success_text("Saved price only 'valuation_scores_full_matrix' DataFrames"))

        # Save the overall scores matrix
        # Price only
        if all_price_columns != []:
            if verbose:
                print(format_info_text("Saving price only 'valuation_scores_full_matrix' DataFrames"))

            try:
                save_dataframe(
                    df=all_score_df[all_price_columns],
                    directory=data_path_config['valuation_scores_full_matrix']['directory'],
                    file_name=data_path_config['valuation_scores_full_matrix']['file_name'],
                    file_format=data_path_config['valuation_scores_full_matrix']['file_format'],
                    save_index=True,
                    versioned=True,
                    suffix="price_only"
                )

            except Exception as e:
                print(format_error_text(f"""Failed to save price only 'valuation_scores_full_matrix': {str(e)}"""))
                raise RuntimeError(format_error_text("Dataframe Save Failed"))


            if verbose:
                print(format_success_text("Saved price only 'valuation_scores_full_matrix' DataFrames"))

        # Save the overall scores matrix
        # Log Price only
        if all_log_price_columns != []:
            if verbose:
                print(format_info_text("Saving log_price only 'valuation_scores_full_matrix' DataFrames"))

            try:
                save_dataframe(
                    df=all_score_df[all_log_price_columns],
                    directory=data_path_config['valuation_scores_full_matrix']['directory'],
                    file_name=data_path_config['valuation_scores_full_matrix']['file_name'],
                    file_format=data_path_config['valuation_scores_full_matrix']['file_format'],
                    save_index=True,
                    versioned=True,
                    suffix="log_price_only"
                )

            except Exception as e:
                print(format_error_text(f"""Failed to save log_price only 'valuation_scores_full_matrix': {str(e)}"""))
                raise RuntimeError(format_error_text("Dataframe Save Failed"))


            if verbose:
                print(format_success_text("Saved log_price only 'valuation_scores_full_matrix' DataFrames"))

        # Save the overall scores matrix
        # Log Returns only
        if all_log_returns_columns != []:
            if verbose:
                print(format_info_text("Saving log_returns only 'valuation_scores_full_matrix' DataFrames"))

            try:
                save_dataframe(
                    df=all_score_df[all_log_returns_columns],
                    directory=data_path_config['valuation_scores_full_matrix']['directory'],
                    file_name=data_path_config['valuation_scores_full_matrix']['file_name'],
                    file_format=data_path_config['valuation_scores_full_matrix']['file_format'],
                    save_index=True,
                    versioned=True,
                    suffix="log_returns_only"
                )

            except Exception as e:
                print(format_error_text(f"""Failed to save log_price only 'valuation_scores_full_matrix': {str(e)}"""))
                raise RuntimeError(format_error_text("Dataframe Save Failed"))


            if verbose:
                print(format_success_text("Saved log_price only 'valuation_scores_full_matrix' DataFrames"))

        # Save the latest scores matrix 
        # All scores
        if verbose:
            print(format_info_text("Saving 'valuation_scores_latest_matrix' DataFrames"))

        try:
            save_dataframe(
                df=latest_score_df,
                directory=data_path_config['valuation_scores_latest_matrix']['directory'],
                file_name=data_path_config['valuation_scores_latest_matrix']['file_name'],
                file_format=data_path_config['valuation_scores_latest_matrix']['file_format'],
                save_index=True,
                versioned=True,
                suffix="all_scores"
            )

        except Exception as e:
            print(format_error_text(f"""Failed to save 'valuation_scores_latest_matrix': {str(e)}"""))
            raise RuntimeError(format_error_text("Dataframe Save Failed"))


        if verbose:
            print(format_success_text("Saved price only 'valuation_scores_latest_matrix' DataFrames"))

        # Save the overall scores matrix
        # Price only
        if all_price_columns != []:
            if verbose:
                print(format_info_text("Saving price only 'valuation_scores_latest_matrix' DataFrames"))

            try:
                save_dataframe(
                    df=latest_score_df[all_price_columns],
                    directory=data_path_config['valuation_scores_latest_matrix']['directory'],
                    file_name=data_path_config['valuation_scores_latest_matrix']['file_name'],
                    file_format=data_path_config['valuation_scores_latest_matrix']['file_format'],
                    save_index=True,
                    versioned=True,
                    suffix="price_only"
                )

            except Exception as e:
                print(format_error_text(f"""Failed to save price only 'valuation_scores_latest_matrix': {str(e)}"""))
                raise RuntimeError(format_error_text("Dataframe Save Failed"))


            if verbose:
                print(format_success_text("Saved price only 'valuation_scores_latest_matrix' DataFrames"))

        # Save the overall scores matrix
        # Log Price only
        if all_log_price_columns != []:
            if verbose:
                print(format_info_text("Saving log_price only 'valuation_scores_latest_matrix' DataFrames"))

            try:
                save_dataframe(
                    df=latest_score_df[all_log_price_columns],
                    directory=data_path_config['valuation_scores_latest_matrix']['directory'],
                    file_name=data_path_config['valuation_scores_latest_matrix']['file_name'],
                    file_format=data_path_config['valuation_scores_latest_matrix']['file_format'],
                    save_index=True,
                    versioned=True,
                    suffix="log_price_only"
                )

            except Exception as e:
                print(format_error_text(f"""Failed to save log_price only 'valuation_scores_latest_matrix': {str(e)}"""))
                raise RuntimeError(format_error_text("Dataframe Save Failed"))


            if verbose:
                print(format_success_text("Saved log_price only 'valuation_scores_latest_matrix' DataFrames"))

        # Save the overall scores matrix
        # Log Returns only
        if all_log_returns_columns != []:
            if verbose:
                print(format_info_text("Saving log_returns only 'valuation_scores_latest_matrix' DataFrames"))

            try:
                save_dataframe(
                    df=latest_score_df[all_log_price_columns],
                    directory=data_path_config['valuation_scores_latest_matrix']['directory'],
                    file_name=data_path_config['valuation_scores_latest_matrix']['file_name'],
                    file_format=data_path_config['valuation_scores_latest_matrix']['file_format'],
                    save_index=True,
                    versioned=True,
                    suffix="log_returns_only"
                )

            except Exception as e:
                print(format_error_text(f"""Failed to save log_price only 'valuation_scores_latest_matrix': {str(e)}"""))
                raise RuntimeError(format_error_text("Dataframe Save Failed"))


            if verbose:
                print(format_success_text("Saved log_price only 'valuation_scores_latest_matrix' DataFrames"))

        if verbose:
            print(format_success_text("All DataFrames saved successfully"))

    if verbose:
        print(format_success_text("Valuation Pipeline Run Successfully."))


    return {
        "metric_df" : metric_df,
        "valuation_score_df": all_score_df
    }   