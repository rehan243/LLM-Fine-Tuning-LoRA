import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        # check if the config file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found at {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"error decoding JSON from {self.config_path}: {str(e)}")
        
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        # get a value from the config, return default if not found
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        # set a value in the config
        self.config[key] = value

    def save(self) -> None:
        # save the config back to the file
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except IOError as e:
            raise IOError(f"error saving config to {self.config_path}: {str(e)}")

# usage example:
# if __name__ == "__main__":
#     config_loader = ConfigLoader('path/to/config.json')
#     config = config_loader.load()
#     print(config)