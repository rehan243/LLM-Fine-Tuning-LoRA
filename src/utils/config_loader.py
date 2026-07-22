import json
import os

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        # check if the config file exists
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            try:
                config = json.load(f)
                return config
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from config file: {e}")

    def get(self, key: str, default=None):
        # get a value from the config, return default if not found
        return self.config.get(key, default)

    def __repr__(self):
        return f"<ConfigLoader(config_path='{self.config_path}')>"

# TODO: maybe add a method to save the config back to file if modified
# TODO: implement support for YAML files in addition to JSON