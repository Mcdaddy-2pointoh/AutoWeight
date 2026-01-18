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
    """not
    Function: Using the `column_name` column of the DataFrame df creates a log price value for the same
    """

    if isinstance(column_names, str):
        df[f'log_{column_names}'] = np.log(df[column_names])

    elif isinstance(column_names, list):
        for column in column_names:
            df[f'log_{column}'] = np.log(df[column])

    df.dropna(inplace=True)

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

    if apply_bounding:
        df[f'bounded_valuation_score_{column_name}'] = np.tanh(score)

    if clip is not None:
        df[f'valuation_score_{column_name}'] = score.clip(lower=clip)

    else: 
        df[f'valuation_score_{column_name}'] = score

    return df

def raw_valuation_score_computation(
    config_path: str | Path,
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
        print(format_warn_text(
            f"Using price metric {valuation_parameters_config['metric']}, the use OHLC prices for long-term valuation is discouraged"
        ))
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

    metric_df["Date"] = pd.to_datetime(metric_df["Date"])
    metric_df.set_index(keys="Date", inplace=True)

    if len(metric_df) < valuation_parameters_config['window_in_days']:
        print(format_error_text("window_in_days parameter in [parameters -> valuation] cannot be greater than the price data frames length."))
        raise RuntimeError(format_error_text("Window provided for aggregation greater than the dataframe"))

    # Rename Column name with suffix price
    metric_df.rename(
        columns={column_nm: f'price_{column_nm}' for column_nm in metric_df.columns},
        inplace=True
    )

    price_columns = [column for column in metric_df.columns if str(column).startswith("price_")]

    method = valuation_parameters_config['method']
    if method != "log_price_z_score":
        print(format_warn_text(
            f"Method '{method}' is deprecated. Using 'log_price_z_score' instead."
        ))

    # Calculate log price
    if metric_df.shape != metric_df.dropna().shape:
        print(format_warn_text("NULLs found in price_df, dropping NULLs before calculating `log_price`"))
        metric_df = metric_df.dropna()

    metric_df = calculate_log_column(df=metric_df, column_names=price_columns)

    z_score_columns = list(set([
        column for column in metric_df.columns
        if str(column).startswith("log_price_")
    ]))

    if verbose:
        print(format_info_text("Computing Valuation scores:"))

    all_score_df = None
    apply_bounding_flag = valuation_parameters_config['apply_bounding']

    for column in z_score_columns:
        if verbose:
            print(format_info_text(f"    Computing valuation score for column: {column}"))

        temp_score_df = calculate_score(
            df=metric_df,
            column_name=column,
            window=valuation_parameters_config['window_in_days'],
            clip=clip,
            apply_bounding=apply_bounding_flag
        )

        if apply_bounding_flag and temp_score_df[f'bounded_valuation_score_{column}'].max() == temp_score_df[f'bounded_valuation_score_{column}'].min():
            print(format_warn_text(f"No variation in valuation score observed for {column}. For a robust calculation re-run the valuation pipeline with percentile_dispersion flag in [parameters -> valuation] as 'True'"))

        if all_score_df is None:
            all_score_df = temp_score_df
        else:
            merge_cols = [f'valuation_score_{column}']
            if apply_bounding_flag:
                merge_cols.append(f'bounded_valuation_score_{column}')

            if not set(all_score_df.columns).issuperset(set(merge_cols)):
                all_score_df = pd.merge(
                    left=all_score_df,
                    right=temp_score_df[merge_cols],
                    right_index=True,
                    left_index=True,
                    how="outer"
                )


    all_score_df.sort_index(inplace=True)
    latest_score_df = all_score_df.iloc[[-1]].copy(deep=True)

    all_score_df.dropna(inplace=True)
    latest_score_df.dropna(inplace=True)

    if save_data:
        save_dataframe(
            df=metric_df,
            directory=data_path_config['valuation_input_matrix']['directory'],
            file_name=data_path_config['valuation_input_matrix']['file_name'],
            file_format=data_path_config['valuation_input_matrix']['file_format'],
            save_index=True,
            versioned=True
        )

        save_dataframe(
            df=all_score_df,
            directory=data_path_config['valuation_scores_full_matrix']['directory'],
            file_name=data_path_config['valuation_scores_full_matrix']['file_name'],
            file_format=data_path_config['valuation_scores_full_matrix']['file_format'],
            save_index=True,
            versioned=True,
            suffix="all_scores"
        )

        save_dataframe(
            df=latest_score_df,
            directory=data_path_config['valuation_scores_latest_matrix']['directory'],
            file_name=data_path_config['valuation_scores_latest_matrix']['file_name'],
            file_format=data_path_config['valuation_scores_latest_matrix']['file_format'],
            save_index=True,
            versioned=True,
            suffix="all_scores"
        )

    if verbose:
        print(format_success_text("Valuation Pipeline Run Successfully."))

    return {
        "metric_df": metric_df,
        "valuation_score_df": all_score_df
    }
