import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch.optim as optim

from VAE_client.vae_run_client import VAERunClient
from VAE_client.dataset.fashion_mnist_dataset import get_fashion_mnist_data
from VAE_model.simple_vae_32to64 import VAE_32to64
from VAE_model.loss_functions import bce_reconstruction_loss
from app.utils import select_run_mode, vae_run_with_selected_mode


if __name__ == "__main__":
    # 保存先のパス
    weights_file_name = "simple_vae_32to64_fashion-mnist.pth"
    model_file_name = "simple_vae_32to64_fashion-mnist.pth"

    # モデルを実行するクライエントの準備
    client = VAERunClient()
    client.set_model(VAE_32to64)

    optimizer = optim.Adam(client.model.parameters(), lr=0.001)
    loss_function = bce_reconstruction_loss
    client.set_loss_function_and_optimizer(loss_function, optimizer)

    batch_size = 32
    epoch = 5
    client.set_data(batch_size, get_fashion_mnist_data)

    # モードの選択と実行
    mode = select_run_mode()
    vae_run_with_selected_mode(
        mode,
        client,
        batch_size,
        epoch,
        weights_file_name,
        model_file_name
    )
