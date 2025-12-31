from typing import Dict, Any, List


class ConfigValidationError(ValueError):
    """Raised when correlation config validation fails."""


def validate_correlation_config(config: Dict[str, Any]) -> None:
    """
    Validate correlation configuration parameters.

    Raises:
        ConfigValidationError
    """

    # --------------------
    # Required top-level keys
    # --------------------
    required_keys = {
        "method",
        "metrics",
        "filter",
        "min_observations",
        "optimization_strategy",
    }

    missing = required_keys - config.keys()
    if missing:
        raise ConfigValidationError(f"Missing keys: {missing}")

    # --------------------
    # method
    # --------------------
    allowed_methods = {"pearson", "kendall", "spearman"}
    method = config["method"]

    if method not in allowed_methods:
        raise ConfigValidationError(
            f"method must be one of {allowed_methods}, got '{method}'"
        )

    # --------------------
    # metrics
    # --------------------
    metrics = config["metrics"]

    if not isinstance(metrics, list) or not metrics:
        raise ConfigValidationError("metrics must be a non-empty list")

    allowed_metrics = {"open", "high", "low", "close", "volume"}
    invalid_metrics = set(metrics) - allowed_metrics

    if invalid_metrics:
        raise ConfigValidationError(
            f"Invalid metrics {invalid_metrics}. Allowed: {allowed_metrics}"
        )

    # --------------------
    # filter block
    # --------------------
    filter_cfg = config["filter"]

    if not isinstance(filter_cfg, dict):
        raise ConfigValidationError("filter must be a dictionary")

    required_filter_keys = {
        "filter_n_pairs",
        "top_n_pairs",
        "filter_inverse_threshold",
        "inverse_threshold",
    }

    missing_filter = required_filter_keys - filter_cfg.keys()
    if missing_filter:
        raise ConfigValidationError(f"Missing filter keys: {missing_filter}")

    if not isinstance(filter_cfg["filter_n_pairs"], bool):
        raise ConfigValidationError("filter_n_pairs must be boolean")

    if not isinstance(filter_cfg["top_n_pairs"], int) or filter_cfg["top_n_pairs"] <= 0:
        raise ConfigValidationError("top_n_pairs must be a positive integer")

    if not isinstance(filter_cfg["filter_inverse_threshold"], bool):
        raise ConfigValidationError("filter_inverse_threshold must be boolean")

    inverse_threshold = filter_cfg["inverse_threshold"]
    if not isinstance(inverse_threshold, (int, float)) or not -1 <= inverse_threshold <= 1:
        raise ConfigValidationError(
            "inverse_threshold must be a number between -1 and 1"
        )

    # --------------------
    # min_observations
    # --------------------
    min_obs = config["min_observations"]

    if not isinstance(min_obs, int) or min_obs <= 1:
        raise ConfigValidationError(
            "min_observations must be an integer > 1"
        )

    # --------------------
    # optimization_strategy
    # --------------------
    allowed_strategies = {"negative", "low"}
    strategy = config["optimization_strategy"]

    if strategy not in allowed_strategies:
        raise ConfigValidationError(
            f"optimization_strategy must be one of {allowed_strategies}, got '{strategy}'"
        )
