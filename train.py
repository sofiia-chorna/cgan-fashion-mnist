import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from utils import get_device


def load_checkpoint(checkpoint_path, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """Load model and optimizer states from a checkpoint file"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}...")
        return start_epoch
    return 0


def save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, g_loss_epoch, d_loss_epoch, checkpoint_dir):
    """Save model and optimizer states along with epoch and loss"""
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "generator_optimizer_state_dict": generator_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
        "gen_loss": g_loss_epoch,
        "disc_loss": d_loss_epoch,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"cgan_checkpoint_epoch_{epoch+1}.pth"))
    print("Checkpoint saved for epoch", epoch+1)


def save_generated_samples(epoch, generator, z_dim, n_classes, samples_dir, device):
    """Generate and save samples from the generator with labels"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, z_dim).to(device)
        labels = torch.randint(0, n_classes, (16,)).to(device)
        fake_images = generator(noise, labels).view(-1, 1, 28, 28).cpu()

        # save labels
        labels_filename = os.path.join(samples_dir, f"labels_epoch_{epoch+1}.txt")
        with open(labels_filename, 'w') as f:
            f.write("Labels for generated samples:\n")
            f.write(str(labels.tolist()))

        # save images
        grid = torch.cat([fake_images[i] for i in range(16)], dim=2)
        plt.imshow(grid.squeeze(), cmap="gray")
        plt.title(f"Generated samples at epoch {epoch+1}")
        plt.axis("off")
        plt.savefig(os.path.join(samples_dir, f"generated_samples_epoch_{epoch+1}.png"))
        plt.close()
    generator.train()


def train_cgan(generator, discriminator, dataloader, params):
    g_losses, d_losses = [], []

    # params
    lr = params.get("learning_rate", 0.1)
    momentum = params.get("momentum", 0.5)
    decay_factor = params.get("decay_factor", 1.00004)
    num_epochs = params.get("num_epochs", 10)
    z_dim = params.get("z_dim", 100)
    n_classes = params.get("n_classes", 10)
    checkpoint_params = params.get("checkpoint", {})
    checkpoint_dir = checkpoint_params.get("output_dir", "checkpoints/")
    checkpoint_save_frequency = checkpoint_params.get("save_frequency", 100)
    samples_params = params.get("generated_samples", {})
    samples_dir = samples_params.get("output_dir", "generated_samples/")
    samples_save_frequency = samples_params.get("save_frequency", 100)
    checkpoint_path = params.get("checkpoint_path")

    # ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    device = get_device()

    bce = nn.BCELoss().to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator_optimizer = SGD(generator.parameters(), lr=lr, momentum=momentum)
    discriminator_optimizer = SGD(discriminator.parameters(), lr=lr, momentum=momentum)

    generator_scheduler = ExponentialLR(generator_optimizer, gamma=decay_factor)
    discriminator_scheduler = ExponentialLR(discriminator_optimizer, gamma=decay_factor)

    # load checkpoint if provided
    start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, generator_optimizer, discriminator_optimizer)

    # training loop
    for epoch in range(start_epoch, num_epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_size = images.size(0)
            real_images = images.view(batch_size, -1).to(device)  # flatten images
            labels = labels.to(device)

            # labels for real (1s) and fake (0s) images
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)


            # ---- train discriminator ----
            discriminator_optimizer.zero_grad()

            # real images
            real_output = discriminator(real_images, labels)

            # fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise, labels)
            fake_output = discriminator(fake_images.detach(), labels)

            # get loss
            disc_loss = (bce(real_output, real_labels) + bce(fake_output, fake_labels)) / 2
            disc_loss.backward()
            discriminator_optimizer.step()


            # ---- train generator ----
            generator_optimizer.zero_grad()

            # get fake images
            fake_output = discriminator(fake_images, labels)

            # get loss
            gen_loss = bce(fake_output, real_labels)
            gen_loss.backward()
            generator_optimizer.step()


            # sum losses
            g_loss_epoch += gen_loss.item()
            d_loss_epoch += disc_loss.item()

        # update learning rate
        generator_scheduler.step()
        discriminator_scheduler.step()

        # average losses for this epoch
        g_losses.append(g_loss_epoch / len(dataloader))
        d_losses.append(d_loss_epoch / len(dataloader))

        print("Gen loss:", g_losses[-1], "Disc loss:", d_losses[-1])

        # save checkpoints and samples periodically
        if (epoch + 1) % checkpoint_save_frequency == 0:
            save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, g_loss_epoch, d_loss_epoch, checkpoint_dir)

        if (epoch + 1) % samples_save_frequency == 0:
            save_generated_samples(epoch, generator, z_dim, n_classes, samples_dir, device)

    # final checkpoint save
    save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, g_loss_epoch, d_loss_epoch, checkpoint_dir)
    save_generated_samples(epoch, generator, z_dim, n_classes, samples_dir, device)
