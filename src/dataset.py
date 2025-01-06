from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_fashion_mnist_dataloader(train=True, batch_size=64, data_dir="../data"):
    """Creates a DataLoader for the FashionMNIST"""
    # load the FashionMNIST dataset
    ds = FashionMNIST(data_dir, train=train, transform=ToTensor(), download=True)

    # wrap the dataset in a dataloader
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloader
