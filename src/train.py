import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from utils import get_device, plot_losses, load_checkpoint


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


def save_generated_samples(epoch, generator, z_dim, n_classes, device):
    """Generate and save samples from the generator with labels"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, z_dim).to(device)
        labels = torch.randint(0, n_classes, (16,)).to(device)
        fake_images = generator(noise, labels).view(-1, 1, 28, 28).cpu()

        # save images
        grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)

        # log to w&b
        wandb.log({
            "generated_samples": wandb.Image(grid, caption=f"Epoch {epoch+1}"),
            "epoch": epoch + 1,
        })
    generator.train()


def train_cgan(generator, discriminator, dataloader, params):
    g_losses, d_losses = [], []

    # init w&b
    wandb.init(project="cgan-fashion-mnist", config=params, job_type="train")
    config = wandb.config

    # params
    lr = config.get("learning_rate", 0.1)
    momentum = config.get("momentum", 0.5)
    decay_factor = config.get("decay_factor", 1.00004)
    num_epochs = config.get("num_epochs", 10)
    z_dim = config.get("z_dim", 100)
    n_classes = config.get("n_classes", 10)
    checkpoint_params = config.get("checkpoint", {})
    checkpoint_dir = checkpoint_params.get("output_dir", "checkpoints/")
    checkpoint_save_frequency = checkpoint_params.get("save_frequency", 100)
    samples_params = config.get("generated_samples", {})
    samples_save_frequency = samples_params.get("save_frequency", 100)
    checkpoint_path = config.get("checkpoint_path")
    loss_plots_dir = os.path.join(checkpoint_dir, "loss_plots/")

    # ensure directory exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(loss_plots_dir, exist_ok=True)

    # setup training
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

        # log to w&b
        wandb.log({
            "epoch": epoch + 1,
            "gen_loss": g_losses[-1],
            "disc_loss": d_losses[-1],
            "lr_generator": generator_scheduler.get_last_lr()[0],
            "lr_discriminator": discriminator_scheduler.get_last_lr()[0],
        })

        print("Gen loss:", g_losses[-1], "Disc loss:", d_losses[-1])

        # save checkpoints and samples periodically
        if (epoch + 1) % checkpoint_save_frequency == 0:
            save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, g_loss_epoch, d_loss_epoch, checkpoint_dir)

        if (epoch + 1) % samples_save_frequency == 0:
            save_generated_samples(epoch, generator, z_dim, n_classes, device)

    # final checkpoint save
    save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, g_loss_epoch, d_loss_epoch, checkpoint_dir)
    save_generated_samples(epoch, generator, z_dim, n_classes, device)

    # finish logging
    wandb.finish()

    plot_losses(g_losses, d_losses, epoch, loss_plots_dir)
