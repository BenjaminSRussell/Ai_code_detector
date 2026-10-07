"""
File processing utilities.

This module contains helper functions for reading and processing files.
"""

import os


def read_file(file_path):
    """
    Read the contents of a file.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        str: The contents of the file, or None if an error occurred.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None
        # Open the file and read its contents
        with open(file_path, "r") as file:
            data = file.read()
        # Return the data
        return data
    except Exception as e:
        # Handle any errors that occur
        print(f"An error occurred while reading the file: {e}")
        return None


def process_data(data):
    """
    Process the given data.

    Args:
        data (str): The data to process.

    Returns:
        list: A list of processed lines.
    """
    # Initialize the result list
    result = []
    # Iterate over each line in the data
    for line in data.split("\n"):
        # Strip whitespace from the line
        processed_line = line.strip()
        # Add the processed line to the result if it is not empty
        if processed_line:
            result.append(processed_line)
    # Return the result
    return result
