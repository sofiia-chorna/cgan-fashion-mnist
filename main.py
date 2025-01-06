from utils import read_yaml
from nn.generator import Generator
from nn.discriminator import Discriminator


yaml_file = "params.yaml"
params = read_yaml(yaml_file)

generator = Generator(params["model"])
discriminator = Discriminator(params["model"])
