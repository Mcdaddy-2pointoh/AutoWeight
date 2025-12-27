from setuptools import setup, find_packages

setup(
    name="autoweight",
    version="0.1.0",
    description="A lightweight, configurable investment-weighting engine that dynamically allocates capital across multiple assets using real-time market data. The tool is designed to compute optimal portfolio weightings based on current market conditions, enabling systematic and repeatable rebalancing. It is particularly useful for long-term investment strategies where periodic, data-driven portfolio rebalancing is required to manage risk and maintain desired exposure across asset cycles.",
    author="Sharvil Dandekar",
    author_email="sharvil.public@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "yahoo_fin",
        "yfinance",
        "pyyaml",
    ],
    python_requires=">=3.12.1",
)