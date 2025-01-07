import os
from datetime import datetime
import torch
import torchvision
import wandb
from utils import load_checkpoint, get_device


def save_generated_samples(generator, output_dir, num_samples):
    # generate random noise
    device = get_device()
    noise = torch.randn(num_samples, generator.z_dim).to(device)
    labels = torch.randint(0, generator.n_classes, (num_samples,)).to(device)
    generated_samples = generator(noise, labels)

    # generate names for files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"generated_samples_{timestamp}.png"
    labels_file_name = f"labels_{timestamp}.txt"

    # save generated images
    grid = torchvision.utils.make_grid(generated_samples.cpu(), nrow=8)
    save_path = os.path.join(output_dir, file_name)
    torchvision.utils.save_image(grid, save_path)

    # save labels
    labels_save_path = os.path.join(output_dir, labels_file_name)
    with open(labels_save_path, 'w') as f:
        for label in labels.cpu().numpy():
            f.write(f"{label} ")

    # log
    wandb.log({
        "generated_samples": wandb.Image(save_path),
        "labels_used": wandb.Table(columns=["Label"], data=[[label] for label in labels.cpu().numpy()])
    })

    return generated_samples, labels


def get_pca():
    pass


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

    # ensure directory exist
    os.makedirs(output_dir, exist_ok=True)

    # load checkpoint if provided
    _ = load_checkpoint(checkpoint_path, generator, discriminator)

    # save some generated samples
    generated_samples, labels = save_generated_samples(generator, output_dir, num_samples)

    # pca

    # fid
