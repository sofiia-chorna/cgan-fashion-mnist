import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from utils.model import load_checkpoint, save_checkpoint, save_generated_samples, calculate_fid
from utils.general import DEVICE, plot_losses, plot_fid
from dim_red import get_feature_extractor


def train_cgan(generator, discriminator, dataloader, params):
    g_losses, d_losses, fid_scores = [], [], []
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
    fid_calculation_frequency = config.get("fid_frequency", 0)
    fid_num_samples = config.get("fid_num_samples", 0)

    # generate run id
    id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # ensure directory exist
    run_dir = os.path.join("../runs", id)
    os.makedirs(run_dir, exist_ok=True)
    plots_dir = os.path.join(run_dir, "plots/")
    os.makedirs(plots_dir, exist_ok=True)
    samples_dir = os.path.join(run_dir, "generated/")
    os.makedirs(samples_dir, exist_ok=True)

    # get pretrained model if fid calculation is necessary
    if fid_calculation_frequency > 0:
        feature_extractor = get_feature_extractor()
        print("Pretrained model for FID calculation is loaded")

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

            # smoothed real and fake labels
            real_labels = torch.ones(batch_size, 1) * 0.9
            fake_labels = torch.zeros(batch_size, 1)
            real_labels = real_labels.to(DEVICE)
            fake_labels = fake_labels.to(DEVICE)

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

        # calculate fid periodically
        if fid_calculation_frequency > 0 and (epoch + 1) % fid_calculation_frequency == 0:
            fid = calculate_fid(generator, dataloader, feature_extractor, num_samples=fid_num_samples)
            fid_scores.append(fid)
            wandb.log({"FID": fid})
            print(f"FID for epoch {epoch+1}: {fid}")

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

    # plot FID scores
    if fid_calculation_frequency > 0:
        fid_plot_path = os.path.join(plots_dir, "fid_scores.png")
        plot_fid(fid_scores, fid_plot_path)
        wandb.log({"FID plot": wandb.Image(fid_plot_path)})

    # finish logging
    wandb.finish()

    plot_losses(g_losses, d_losses, epoch, plots_dir)
