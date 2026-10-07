"""
Data validation module.

This module provides functions for validating input data.
"""


def validate_email(email):
    """
    Validate an email address.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    # Check if the email contains an @ symbol
    if "@" not in email:
        return False
    # Split the email into parts
    parts = email.split("@")
    # Check if there are exactly two parts
    if len(parts) != 2:
        return False
    # Return True if all checks pass
    return True


def validate_data(data):
    """
    Validate the given data.

    Args:
        data (dict): The data to validate.

    Returns:
        bool: True if the data is valid, False otherwise.
    """
    try:
        # Check if the data is a dictionary
        if not isinstance(data, dict):
            return False
        # Check if the required fields are present
        for field in ["name", "email"]:
            if field not in data:
                return False
        # Validate the email field
        return validate_email(data["email"])
    except Exception as e:
        # Print an error message
        print(f"Validation error: {e}")
        return False
