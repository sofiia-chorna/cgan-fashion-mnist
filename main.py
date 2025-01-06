from utils import read_yaml
from nn.generator import Generator
from nn.discriminator import Discriminator
from dataset import get_fashion_mnist_dataloader
from train import train_cgan


yaml_file = "params.yaml"
params = read_yaml(yaml_file)

generator = Generator(params["model"])
discriminator = Discriminator(params["model"])

train_dl = get_fashion_mnist_dataloader(train=True)

train_cgan(generator, discriminator, train_dl, params["training"])
