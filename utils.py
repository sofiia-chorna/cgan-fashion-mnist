import torch
import yaml


def read_yaml(file_path):
    """
    file_path: the path to the YAML file
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device
