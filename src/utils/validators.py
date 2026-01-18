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
        "filter",
        "min_observations",
        "optimization_strategy",
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

    # Configurations for filters applied must be suitable
    filter_cfg = config["filter"]

    if not isinstance(filter_cfg, dict):
        raise ConfigValidationError("filter in [parameters -> correlation] must be a dictionary")

    required_filter_keys = {
        "filter_n_pairs",
        "top_n_pairs",
    }

    missing_filter = required_filter_keys - filter_cfg.keys()
    if missing_filter:
        raise ConfigValidationError(f"Missing filter keys in [parameters -> correlation]: {missing_filter}")

    if not isinstance(filter_cfg["filter_n_pairs"], bool):
        raise ConfigValidationError("filter_n_pairs in [parameters -> correlation -> filter] must be boolean")

    if not isinstance(filter_cfg["top_n_pairs"], int) or filter_cfg["top_n_pairs"] <= 0:
        raise ConfigValidationError("top_n_pairs in [parameters -> correlation -> filter] must be a positive integer")

    # Minimum obsvs needed for calculating corr must be specified
    min_obs = config["min_observations"]

    if not isinstance(min_obs, int) or min_obs <= 1:
        raise ConfigValidationError(
            "min_observations in [parameters -> correlation] must be an integer > 1"
        )

    # Optimization strategy must be a categorical var from the following opts
    allowed_strategies = {"negative", "low"}
    strategy = config["optimization_strategy"]

    if strategy not in allowed_strategies:
        raise ConfigValidationError(
            f"optimization_strategy in [parameters -> correlation] must be one of {allowed_strategies}, got '{strategy}'"
        )


def validate_valuation_config(config: Dict[str, Any]) -> None:
    """
    Validate valuation configuration parameters.

    Raises:
        ConfigValidationError
    """

    # Top Level Key validations
    required_keys = {
        "window_in_days",
        "metric",
        "method",
        "apply_bounding"
    }

    missing = required_keys - config.keys()
    if missing:
        raise ConfigValidationError(f"Missing keys in [parameters -> valuation]: {missing}")

    # If the window_in_days is not an integer Raise an error
    window_in_days = config['window_in_days']
    if not isinstance(window_in_days, int) or not window_in_days > 0:
        raise ConfigValidationError(
            f"window_in_days in [parameters -> valuation] must be an 'int' type value greater than 1, got '{window_in_days}' with type {type(window_in_days)}"
        )

    # If the price metric is not close warn about the implications
    metric = config['metric']

    # Price metrics must be based out of the ones defined in yfinance

    if not isinstance(metric, str) or not metric:
        raise ConfigValidationError("metric in [parameters -> valuation] must be a non-empty string")

    allowed_metrics = {"open", "high", "low", "close", "adj_close"}

    if metric not in allowed_metrics:
        raise ConfigValidationError(
            f"Invalid metric {metric} in [parameters -> valuation]. Allowed: {allowed_metrics}"
        )

    # If the method of valuation is not in the defined methods raise errors
    allowed_methods = ['price_z_score', 'log_price_z_score', 'log_return_z_score', 'all']
    method = config['method']

    if not isinstance(method, str) or not method:
        raise ConfigValidationError("method in [parameters -> valuation] must be a non-empty string")
        
    if not method in allowed_methods:
        raise ConfigValidationError(
            f"Invalid method {method} in [parameters -> valuation]. Allowed: {allowed_methods}"
        ) 
    
    
    # If the apply_bounding flag is not defined
    apply_bounding_flag = config['apply_bounding']
    if not isinstance(apply_bounding_flag, bool):
        raise ConfigValidationError("apply_bounding_flag in [parameters -> valuation] must be a boolean")
    

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
    
    


    


    