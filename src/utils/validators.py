def validate_correlation_config(config: dict) -> None:
    """
    Validate correlation configuration dictionary.

    Expected schema:
        {
            "method": str in {"pearson", "kendall", "spearman"},
            "metrics": str | list[str] subset of allowed metrics,
            "inverse_threshold": float between -1 and 1
        }

    Raises:
        ValueError: If validation fails
    """

    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config).__name__}")

    # Validate the method param
    allowed_methods = {"pearson", "kendall", "spearman"}
    method = config.get("method")

    if not isinstance(method, str):
        raise ValueError("`method` must be a string")

    if method not in allowed_methods:
        raise ValueError(
            f"`method` must be one of {sorted(allowed_methods)}, got '{method}'"
        )

    # Validate the metric param
    allowed_metrics = {"open", "high", "low", "close", "volume"}
    metrics = config.get("metrics")

    if isinstance(metrics, str):
        metrics = [metrics]
    elif not isinstance(metrics, list):
        raise ValueError("`metrics` must be a string or a list of strings")

    if not metrics:
        raise ValueError("`metrics` cannot be empty")

    if not all(isinstance(m, str) for m in metrics):
        raise ValueError("All values in `metrics` must be strings")

    invalid_metrics = set(metrics) - allowed_metrics
    if invalid_metrics:
        raise ValueError(
            f"Invalid metrics {sorted(invalid_metrics)}. "
            f"Allowed metrics are {sorted(allowed_metrics)}"
        )

    # Validate the inverse threshold value
    inverse_threshold = config.get("inverse_threshold")

    if not isinstance(inverse_threshold, float):
        raise ValueError("`inverse_threshold` must be a float")

    if not -1.0 <= inverse_threshold <= 1.0:
        raise ValueError("`inverse_threshold` must be between -1 and 1")

    # ---- success ----
    return True
