class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def format_error_text(text: str) -> str:
    """
    Function: Produces a red bold text with the inpu text
    Args:
        text (str): The text to color as a ERROR
    """

    return f"""{bcolors.FAIL}{bcolors.BOLD}ERROR: {str(text)}{bcolors.ENDC}"""

def format_success_text(text: str) -> str:
    """
    Function: Produces a green bold text with the input text
    Args:
        text (str): The text to color as a SUCCESS
    """

    return f"""{bcolors.BOLD}{bcolors.OKGREEN}SUCCESS: {str(text)}{bcolors.ENDC}"""

def format_info_text(text: str) -> str:
    """
    Function: Produces a yellow text with the input text
    Args:
        text (str): The text to color as a INFO
    """

    return f"""{bcolors.WARNING}INFO: {str(text)}{bcolors.ENDC}"""