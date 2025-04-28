import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from utils import (
    get_device, 
    get_fashion_mnist_dataset
)


if __name__ == "__main__":
    # デバイスの取得
    device = get_device()

    # データの取得
    BATCH_SIZE = 4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_set, validation_set = get_fashion_mnist_dataset(transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

    classes = (
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    )

    