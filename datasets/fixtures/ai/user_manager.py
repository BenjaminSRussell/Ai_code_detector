"""
User management module.

This module provides functionality for managing users in the system.
"""


class UserManager:
    """
    A class to manage users.

    This class provides methods to add, remove, and retrieve users.
    """

    def __init__(self):
        """Initialize the UserManager with an empty list of users."""
        # Initialize the users list
        self.users = []

    def add_user(self, user_data):
        """
        Add a new user to the system.

        Args:
            user_data: The data of the user to add.

        Returns:
            bool: True if the user was added successfully, False otherwise.
        """
        try:
            # Check if the user data is valid
            if user_data is None:
                return False
            # Add the user to the list
            self.users.append(user_data)
            return True
        except Exception as e:
            # Print the error message
            print(f"An error occurred: {e}")
            return False

    def get_user(self, user_id):
        """
        Retrieve a user by their ID.

        Args:
            user_id: The ID of the user to retrieve.

        Returns:
            The user data if found, None otherwise.
        """
        # Iterate through the users list
        for user in self.users:
            # Check if the user ID matches
            if user.get("id") == user_id:
                return user
        # Return None if the user was not found
        return None
