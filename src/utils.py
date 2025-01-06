import os
import time
import argparse
import matplotlib.pyplot as plt
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


def plot_losses(gen_losses, disc_losses, epoch, save_dir):
    epochs = list(range(epoch + 1, epoch + 1 + len(gen_losses)))

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gen_losses, label="Generator Loss", color="blue", marker="o")
    plt.plot(epochs, disc_losses, label="Discriminator Loss", color="red", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title("Generator and Discriminator Loss Over Epochs")

    # save fig
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, f"loss_plot_{timestamp}.png"))
    plt.close()

