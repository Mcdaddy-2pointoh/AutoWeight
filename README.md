# AutoWeight (Work In Progress)
A lightweight, configurable investment-weighting engine that dynamically allocates capital across multiple assets using real-time market data. The tool is designed to compute optimal portfolio weightings based on current market conditions, enabling systematic and repeatable rebalancing. It is particularly useful for long-term investment strategies where periodic, data-driven portfolio rebalancing is required to manage risk and maintain desired exposure across asset cycles.

# How to Clone & Setup the Repo
## 1. Clone the Repo:
At the desired local path clone the repo using `git clone https://github.com/Mcdaddy-2pointoh/AutoWeight.git`.

## 2. Initialise a Python VENV:
Ensure you have python >=3.12.1 and create a local virtual environment using `python -m venv <your-venv-name>`

## 3. Create the necessary folder structure:
Within the repository create a folder called data
    1. Check the current working directory (cwd)
    2. Create a folder within the current working directory using `mkdir <cwd>/data`
    3. Run the `initialize.py` file and provide the path `<cwd>/data` when prompted for input
    4. Directory Structure for Data Created

## 4. Configure `config.yml`:
Replace the key term `basepath` in config.yml with the data location source `<cwd>/data`

## 5. Install Additional Dependecies:
Run the command `pip install -e .` to load the repo as a library

## 6. Setup Complete! :D

