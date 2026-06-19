import json
import os
from typing import Any, Dict

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def save(self) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

# usage example:
# if __name__ == "__main__":
#     config_loader = ConfigLoader('path/to/config.json')
#     config = config_loader.load()
#     print(config)