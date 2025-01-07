import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from utils import DEVICE, plot_losses


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


def train_cgan(generator, discriminator, dataloader, params):
    g_losses, d_losses = [], []
    best_g_loss, best_d_loss = float('inf'), float('inf')

    # init w&b
    wandb.init(project="cgan-fashion-mnist", config=params, job_type="train")
    config = wandb.config

    # params
    lr = config.get("learning_rate", 0.1)
    momentum = config.get("momentum", 0.5)
    decay_factor = config.get("decay_factor", 1.00004)
    num_epochs = config.get("num_epochs", 10)
    z_dim = config.get("z_dim", 100)
    samples_save_frequency = config.get("samples_save_frequency", 100)
    checkpoint_path = config.get("checkpoint_path")

    # generate run id
    id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # ensure directory exist
    run_dir = os.path.join("../runs", id)
    os.makedirs(run_dir, exist_ok=True)
    loss_plots_dir = os.path.join(run_dir, "loss_plots/")
    os.makedirs(loss_plots_dir, exist_ok=True)
    samples_dir = os.path.join(run_dir, "generated/")
    os.makedirs(samples_dir, exist_ok=True)

    # setup training
    bce = nn.BCELoss().to(DEVICE)
    generator = generator.to(DEVICE)
    discriminator = discriminator.to(DEVICE)

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
            real_images = images.view(batch_size, -1).to(DEVICE)  # flatten images
            labels = labels.to(DEVICE)

            # labels for real (1s) and fake (0s) images
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)


            # ---- train discriminator ----
            discriminator_optimizer.zero_grad()

            # real images
            real_output = discriminator(real_images, labels)

            # fake images
            noise = torch.randn(batch_size, z_dim).to(DEVICE)
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

        # save samples periodically
        if (epoch + 1) % samples_save_frequency == 0:
            save_generated_samples(epoch, generator, samples_dir)

        # save best generator
        if g_losses[-1] < best_g_loss:
            best_g_loss = g_losses[-1]
            torch.save(generator.state_dict(), os.path.join(run_dir, "best_generator.pth"))

        # save best discriminator
        if d_losses[-1] < best_d_loss:
            best_d_loss = d_losses[-1]
            torch.save(discriminator.state_dict(), os.path.join(run_dir, "best_discriminator.pth"))

    # final checkpoint save
    save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, g_loss_epoch, d_loss_epoch, run_dir)
    save_generated_samples(epoch, generator, samples_dir)

    # finish logging
    wandb.finish()

    plot_losses(g_losses, d_losses, epoch, loss_plots_dir)
