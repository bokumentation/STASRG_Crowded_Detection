# File: config_writer.py

import json
import os

CONFIG_FILE = "config.json"

def save_config_to_file(new_data):
    """
    Safely writes the configuration dictionary back to config.json.
    
    Args:
        new_data (dict): The complete dictionary structure to save.
    
    Returns:
        bool: True on success.
    """
    try:
        # 1. Write to a temporary file first for safety
        temp_file = CONFIG_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(new_data, f, indent=4)
            
        # 2. Rename/overwrite the original file only if writing was successful
        os.replace(temp_file, CONFIG_FILE)
        return True
    except Exception as e:
        print(f"ERROR writing config file: {e}")
        return False