"""
Configuration loader module.

This module provides a class for loading and managing configuration settings.
"""

import json


class ConfigLoader:
    """
    A class to load and manage configuration settings.
    """

    def __init__(self, config_path):
        """
        Initialize the ConfigLoader.

        Args:
            config_path (str): The path to the configuration file.
        """
        # Store the configuration path
        self.config_path = config_path
        # Initialize the configuration dictionary
        self.config = {}

    def load_config(self):
        """
        Load the configuration from the file.

        Returns:
            dict: The loaded configuration.
        """
        try:
            # Open the configuration file
            with open(self.config_path, "r") as f:
                # Load the JSON data
                self.config = json.load(f)
            # Return the configuration
            return self.config
        except Exception as e:
            # Print an error message
            print(f"Failed to load configuration: {e}")
            return {}

    def get_value(self, key, default_value=None):
        """
        Get a value from the configuration.

        Args:
            key (str): The key to look up.
            default_value: The default value to return if the key is not found.

        Returns:
            The value associated with the key, or the default value.
        """
        # Return the value from the configuration
        return self.config.get(key, default_value)
