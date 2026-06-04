import yaml
import os

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {self.config_path}") from e
        
        return config

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def __repr__(self):
        return f"<ConfigLoader config_path={self.config_path}>"

# example usage
if __name__ == "__main__":
    # TODO: replace with actual config file path
    config_loader = ConfigLoader('path/to/config.yaml')
    print(config_loader.get('some_key', 'default_value'))