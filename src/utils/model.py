import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from scipy.linalg import sqrtm
import wandb
from .general import DEVICE
from dim_red import get_embeddings


def initialize_weights(m):
    """Use of xavier initialization"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

        # set biases to 0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def save_checkpoint(epoch, g, d, g_optimizer, d_optimizer, g_loss_epoch, d_loss_epoch, run_dir):
    """Save model, optimizer states, epoch and loss"""
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": g.state_dict(),
        "discriminator_state_dict": d.state_dict(),
        "generator_optimizer_state_dict": g_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": d_optimizer.state_dict(),
        "gen_loss": g_loss_epoch,
        "disc_loss": d_loss_epoch,
    }
    torch.save(checkpoint, os.path.join(run_dir, f"cgan_checkpoint_epoch_{epoch+1}.pth"))
    print("Checkpoint saved for epoch", epoch+1)


def load_checkpoint(checkpoint_path, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """Load model and optimizer states from a checkpoint file."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loading a checkpoint from epoch {start_epoch}...")
        return start_epoch
    return 0


def save_generated_samples(epoch, generator, samples_dir, num_samples=16):
    """Generate and save samples from the generator with labels"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, generator.z_dim).to(DEVICE)
        labels = torch.randint(0, generator.n_classes, (num_samples,)).to(DEVICE)
        fake_images = generator(noise, labels).view(-1, 1, 28, 28).cpu()

        # save images
        grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
        image_save_path = os.path.join(samples_dir, f"epoch_{epoch+1}.png")
        torchvision.utils.save_image(grid, image_save_path)

        # log to w&b
        wandb.log({
            "generated_samples": wandb.Image(grid, caption=f"Epoch {epoch+1}"),
            "epoch": epoch + 1,
        })
    generator.train()


def calculate_fid(generator, dataloader, feature_extractor, num_samples=100):
    """Get FID score for the generator output compared to real data"""
    generator.eval()
    real_features = []
    fake_features = []

    with torch.no_grad():
        # get real features
        for real_images, _real_labels in tqdm(dataloader, desc="FID: extracting real features"):
            real_features.append(get_embeddings(feature_extractor, real_images))
            if len(real_features) * real_images.size(0) >= num_samples:
                break
        real_features = np.vstack(real_features)[:num_samples]

        # generate fake samples and get features
        tqdm.write("FID: generating fake images and extracting features...")
        noise = torch.randn(num_samples, generator.z_dim).to(DEVICE)
        labels = torch.randint(0, generator.n_classes, (num_samples,)).to(DEVICE)
        fake_images = generator(noise, labels).cpu()
        fake_features = get_embeddings(feature_extractor, fake_images)

    # compute statistiques for real and fake features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # get fid score
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    # handle cases where sqrtm produces complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    generator.train()
    return fid
