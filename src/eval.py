import os
from datetime import datetime
import torch
import torchvision
import wandb
from utils import DEVICE, load_checkpoint
from dim_red import run_dim_reduction, get_feature_extractor


def save_images(images, labels, output_dir, prefix, nrow=10):
    """Helper function to save images and labels"""
    # save image grid
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    image_save_path = os.path.join(output_dir, f"{prefix}_images.png")
    torchvision.utils.save_image(grid, image_save_path)

    # save labels
    labels_save_path = os.path.join(output_dir, f"{prefix}_labels.txt")
    with open(labels_save_path, 'w') as f:
        for label in labels.cpu().numpy():
            f.write(f"{label} ")

    # log
    wandb.log({
        f"{prefix}_images_grid": wandb.Image(image_save_path),
        f"{prefix}_labels": wandb.Table(columns=["Label"], data=[[label] for label in labels.cpu().numpy()])
    })

    return image_save_path, labels_save_path


def generate_samples(generator, output_dir, num_samples, save=False):
    # generate random noise
    noise = torch.randn(num_samples, generator.z_dim).to(DEVICE)
    labels = torch.randint(0, generator.n_classes, (num_samples,)).to(DEVICE)
    generated_samples = generator(noise, labels)

    # save generated images
    if save:
        save_images(generated_samples.cpu(), labels, output_dir, "generated")

    return generated_samples, labels


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

    # generate a batch of images
    generated_samples, labels = generate_samples(generator, output_dir, num_samples, save=True)

    # get a batch from the test data
    real_samples, real_labels = next(iter(dataloader))
    save_images(real_samples[:num_samples], real_labels[:num_samples], output_dir, "real")

    # apply dimentionality reduction on real and generated images
    feature_extractor = get_feature_extractor()
    run_dim_reduction(feature_extractor, real_samples, real_labels, output_dir, real=True)
    run_dim_reduction(feature_extractor, generated_samples, labels, output_dir, real=False)

    # fid
