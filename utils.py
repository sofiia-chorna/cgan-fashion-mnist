import yaml


def read_yaml(file_path):
    """
    file_path: the path to the YAML file
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
