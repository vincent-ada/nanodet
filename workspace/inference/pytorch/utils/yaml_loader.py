import yaml
from box import Box


def load_yaml(yaml_path, **box_options):
    try:
        with open(yaml_path, "r") as f:
            data_dict = yaml.safe_load(f)
            return Box(data_dict, **box_options)
    except FileNotFoundError:
        print(f"Error: File not found at {yaml_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")
        return None
