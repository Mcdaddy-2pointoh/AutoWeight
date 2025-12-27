import os 
import pandas as pd
import yaml 
from pathlib import Path
import logging
from src.utils.file_operators import load_yaml

# function to initialise the directory structure
def create_directory_structure(path: str):
    """
    Function: Accepts a path to a base directory and creates all intermediate folders
    Args:
        path (str | Path): The path to the base directory
    """

    # Convert the string to path
    path = Path(path)
    print(f"INFO: Creating Base Directory")

    # Check if directory exists
    if path.exists():
        print(f"Base Directory path exists: {str(path)}")

    # Create directory + all intermediate parents (safe)
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Base Directory path created at: {str(path)}")

    print(f"Creating Sub Directories")

    # Create sub directories
    sub_directories = [

        # Raw Directories
        "01_raw",
        "02_processed",
        "03_analysis"
    ]

    # for sub directories 
    # Create all child directories
    for sub_directory in sub_directories:

        # Create a temp path to create sub directories
        temp_path = path / sub_directory

        # Create the sub directory
        if temp_path.exists():
            print(f"Sub Directory path exists: {str(temp_path)}")

        # Create directory + all intermediate parents (safe)
        else:
            temp_path.mkdir(parents=True, exist_ok=True)
            print(f"Sub Directory path created at: {str(temp_path)}")

    # Logger for successful directory
    print(f"Created all directories successfully")

# Run only if called explicitly from the file
if __name__ == "__main__":

    # Path input
    data_path = str(input("Please enter the path to the base directory: "))
    print(f"Creating Directory Structure at: {data_path}")

    # Create directroy structure
    create_directory_structure(path=data_path)