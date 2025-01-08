import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import yaml

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_yaml(file_path):
    """
    file_path: the path to the YAML file
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def get_input_params():
    """Parse command line arguments to get the path to the params.yaml and mode"""
    parser = argparse.ArgumentParser(description="Train or evaluate CGAN")

    # argument for params yaml
    parser.add_argument(
        "--params",
        type=str,
        default="params.yaml",
        help="path to the parameters yaml file",
    )

    # exclusive arguments for train or eval
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="train model",
    )
    mode_group.add_argument(
        "--eval",
        action="store_true",
        help="evaluate model",
    )

    args = parser.parse_args()

    # return both the params path and selected mode
    mode = 'train' if args.train else 'eval'
    return args.params, mode


def plot_losses(gen_losses, disc_losses, epoch, save_dir):
    epochs = list(range(epoch + 1))

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gen_losses, label="Generator loss", color="blue")
    plt.plot(epochs, disc_losses, label="Discriminator loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title("Generator and Discriminator loss over epochs")

    # save fig
    plt.savefig(os.path.join(save_dir, f"loss.png"))
    plt.close()


def plot_fid(fid_scores, output_path):
    """Plot FID scores"""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(fid_scores) + 1), fid_scores, linestyle='-', color='b')
    plt.title("FID Scores over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("FID score")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
