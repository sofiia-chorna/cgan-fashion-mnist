from cgan.generator import Generator
from cgan.discriminator import Discriminator
from dataset import get_fashion_mnist_dataloader
from train import train_cgan
from eval import eval_cgan
from utils.general import get_input_params, read_yaml
from utils.model import initialize_weights


yaml_file, mode = get_input_params()
params = read_yaml(yaml_file)
batch_size = params.get("batch_size", 64)

generator = Generator(params["model"])
discriminator = Discriminator(params["model"])


if mode == "train":
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    train_dl = get_fashion_mnist_dataloader(train=True, batch_size=batch_size)
    train_cgan(generator, discriminator, train_dl, params["training"])
else:
    test_dl = get_fashion_mnist_dataloader(train=False, batch_size=batch_size)
    eval_cgan(generator, discriminator, test_dl, params["evaluation"])
