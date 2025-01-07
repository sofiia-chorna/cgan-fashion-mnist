import os
from datetime import datetime
import torch
import torchvision
import wandb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import load_checkpoint, get_device


def save_generated_samples(generator, output_dir, num_samples):
    # generate random noise
    device = get_device()
    noise = torch.randn(num_samples, generator.z_dim).to(device)
    labels = torch.randint(0, generator.n_classes, (num_samples,)).to(device)
    generated_samples = generator(noise, labels)

    # save generated images
    grid = torchvision.utils.make_grid(generated_samples.cpu(), nrow=8)
    save_path = os.path.join(output_dir, "generated_samples.png")
    torchvision.utils.save_image(grid, save_path)

    # save labels
    labels_save_path = os.path.join(output_dir, "labels.txt")
    with open(labels_save_path, 'w') as f:
        for label in labels.cpu().numpy():
            f.write(f"{label} ")

    # log
    wandb.log({
        "generated_samples": wandb.Image(save_path),
        "labels_used": wandb.Table(columns=["Label"], data=[[label] for label in labels.cpu().numpy()])
    })

    return generated_samples, labels


def get_pca(samples, labels, output_dir, real=False):
    """Apply PCA on the samples"""
    prefix = 'real' if real else 'generated'

    # flatten to (num_samples, num_features)
    samples = samples.view(samples.size(0), -1).cpu().detach().numpy()

    # run dim reduction with pca
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(samples)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels.cpu().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f"PCA of {prefix} samples")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    pca_plot_path = os.path.join(output_dir, f"{prefix}_pca_plot.png")
    plt.savefig(pca_plot_path)
    plt.close()

    wandb.log({"PCA Plot": wandb.Image(pca_plot_path)})
    print(f"PCA plot saved at {pca_plot_path}")


def get_fid():
    pass


def eval_cgan(generator, discriminator, dataloader, params):
    # init w&b
    wandb.init(project="cgan-fashion-mnist", config=params, job_type="eval")
    config = wandb.config

    # params
    checkpoint_path = config.get("checkpoint_path")
    output_dir = config.get("output_dir")
    num_samples = config.get("num_samples", 64)

    # load checkpoint if provided
    _ = load_checkpoint(checkpoint_path, generator, discriminator)

    # generate run id
    id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(output_dir, id)
    os.makedirs(output_dir, exist_ok=True)

    # save some generated samples
    generated_samples, labels = save_generated_samples(generator, output_dir, num_samples)

    # get a batch from the test data
    real_samples, real_labels = next(iter(dataloader))

    # pca on generated and read samples
    get_pca(generated_samples, labels, output_dir, real=False)
    get_pca(real_samples, real_labels, output_dir, real=True)

    # fid
