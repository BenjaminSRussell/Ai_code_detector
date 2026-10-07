"""
Simple calculator module.

This module provides basic arithmetic operations.
"""


def add(a, b):
    """
    Add two numbers together.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of the two numbers.
    """
    # Return the sum of a and b
    return a + b


def divide(a, b):
    """
    Divide one number by another.

    Args:
        a (float): The numerator.
        b (float): The denominator.

    Returns:
        float: The result of the division, or None if division by zero.
    """
    try:
        # Perform the division
        result = a / b
        # Return the result
        return result
    except Exception as e:
        # Print an error message if division fails
        print(f"Error: {e}")
        return None


def calculate_average(numbers):
    """
    Calculate the average of a list of numbers.

    Args:
        numbers (list): A list of numbers.

    Returns:
        float: The average of the numbers.
    """
    # Check if the list is empty
    if len(numbers) == 0:
        return 0
    # Calculate the sum of the numbers
    total = sum(numbers)
    # Calculate and return the average
    return total / len(numbers)
