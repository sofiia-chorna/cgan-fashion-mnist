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


def load_checkpoint(checkpoint_path, generator, discriminator, generator_optimizer=None, discriminator_optimizer=None, load_optimizers=True):
    """Load model and optimizer states from a checkpoint file."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if load_optimizers:
            if generator_optimizer and discriminator_optimizer:
                generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
                discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        print(f"Loading a checkpoint from epoch {start_epoch}...")
        return start_epoch
    print(f"Checkpoint file {checkpoint_path} is not found")
    return 0


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
