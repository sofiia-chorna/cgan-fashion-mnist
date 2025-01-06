import argparse
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
    """Takes the device for the computation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def get_params_path():
    """Parse command line arguments to get the path to the params.yaml"""
    parser = argparse.ArgumentParser(description="Train CGAN")
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="path to the parameters yaml file",
    )
    args = parser.parse_args()
    return args.params
