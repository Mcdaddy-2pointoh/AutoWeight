from typing import Dict, Any, List

class ConfigValidationError(ValueError):
    """Raised when config validation fails."""


def validate_pairwise_correlation_config(config: Dict[str, Any]) -> None:
    """
    Validate correlation configuration parameters.

    Raises:
        ConfigValidationError
    """

    # Top Level Key validations
    required_keys = {
        "method",
        "metrics",
        "min_observations",
    }

    missing = required_keys - config.keys()
    if missing:
        raise ConfigValidationError(f"Missing keys in [parameters -> correlation]: {missing}")

    # Method of correlation must be from the listed corr methods in pandas
    allowed_methods = {"pearson", "kendall", "spearman"}
    method = config["method"]

    if method not in allowed_methods:
        raise ConfigValidationError(
            f"method in [parameters -> correlation] must be one of {allowed_methods}, got '{method}'"
        )

    # Price metrics must be based out of the ones defined in yfinance
    metrics = config["metrics"]

    if not isinstance(metrics, list) or not metrics:
        raise ConfigValidationError("metrics in [parameters -> correlation] must be a non-empty list")

    allowed_metrics = {"open", "high", "low", "close", "adj_close"}
    invalid_metrics = set(metrics) - allowed_metrics

    if invalid_metrics:
        raise ConfigValidationError(
            f"Invalid metrics {invalid_metrics} in [parameters -> correlation]. Allowed: {allowed_metrics}"
        )

    # Minimum obsvs needed for calculating corr must be specified
    min_obs = config["min_observations"]

    if not isinstance(min_obs, int) or min_obs <= 1:
        raise ConfigValidationError(
            "min_observations in [parameters -> correlation] must be an integer > 1"
        )

def validate_create_portfolio_universe_config(config: Dict[str, Any]) -> None:
    """
    Function: Validates the portfolio universe configs

    Raises:
        ConfigValidationError
    """

    # Required Keys
    required_keys = {
        "correlation_anchor_points",
        "max_bucket_size"
    }

    missing = required_keys - config.keys()

    if missing:
        raise ConfigValidationError(f"Missing keys in [parameters -> portfolio_params]: {missing}")
    
    if (config['correlation_anchor_points'] < 1) or type(config['correlation_anchor_points']) != int:
        raise ConfigValidationError("correlation_anchor_points in [parameters -> portfolio_params] must be an integer greater than 1")
    
    if (config['max_bucket_size'] is not None) and (config['max_bucket_size'] < 1 or type(config['max_bucket_size']) != int):
        raise ConfigValidationError("max_bucket_size in [parameters -> portfolio_params] must be an integer greater than 1 or NULL")
    
    


    


    