import numpy as np
import logging
import json # Added
import os # Added
import copy # Added

# Configure logging - check if handlers are already present
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_position_free(pos, radius, dish_center, dish_radius, objects, min_distance):
    dist_to_center = np.linalg.norm(pos - dish_center)
    if dist_to_center > dish_radius - radius:
        return False
    for obj in objects:
        dist = np.linalg.norm(pos - obj.pos)
        if dist < radius + obj.radius + min_distance:
            return False
    return True

def load_user_training_settings(default_ts, user_settings_file_path):
    loaded_ts = copy.deepcopy(default_ts) # Start with defaults
    if os.path.exists(user_settings_file_path):
        try:
            with open(user_settings_file_path, 'r') as f:
                user_config = json.load(f)

            for key, value_from_file in user_config.items():
                if key in loaded_ts:
                    default_value_type = type(loaded_ts[key])
                    value_from_file_type = type(value_from_file)

                    if default_value_type == value_from_file_type:
                        loaded_ts[key] = value_from_file
                    # Allow int to be loaded as float and vice-versa for numeric settings
                    elif isinstance(loaded_ts[key], (int, float)) and isinstance(value_from_file, (int, float)):
                        loaded_ts[key] = default_value_type(value_from_file) # Cast to original default type
                    else:
                        logging.warning(
                            f"Type mismatch for setting '{key}' in {user_settings_file_path}. "
                            f"Expected {default_value_type.__name__}, got {value_from_file_type.__name__}. Using default."
                        )
                else:
                    logging.warning(f"Unknown setting '{key}' in {user_settings_file_path}. Ignoring.")
            logging.info(f"Loaded user training settings from {user_settings_file_path}")
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error loading or parsing {user_settings_file_path}: {e}. Using default training settings.")
    return loaded_ts

def prettify_setting_key(key_str):
    """Converts a snake_case key string to Title Case with spaces."""
    return key_str.replace('_', ' ').title()