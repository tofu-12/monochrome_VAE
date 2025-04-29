import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_device():
    """
    使用するデバイスを取得

    Returns:
        device (mps or cpu)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device is available and being used.")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, using CPU instead.")
    
    return device


def get_fashion_mnist_dataset(transform):
    """
    Fashion mnistデータセットを取得

    Args:
        transform: データ変形用のtransform

    Returns:
        tuple: (train, validation)
    """
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    return training_set, validation_set


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
