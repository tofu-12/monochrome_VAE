import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from run_client.client.vae_client import VAEClient
from model.VAE.vae_512_to_8 import VAE
from run_client.dataset.pic_512_dataset import get_pic512_data


if __name__ == "__main__":
    # 保存先のパス
    weights_file_path = os.path.join(os.path.dirname(__file__), os.pardir, "model", "weights_file", "pic_512_to_8_weights.pth")
    model_file_path = os.path.join(os.path.dirname(__file__), os.pardir, "model", "model_file", "pic_512_to_8_model.pth")

    is_train = False

    client = VAEClient(VAE)
    if is_train:
        client.run_training(250, 20, get_pic512_data, weights_file_path, model_file_path, False)
    else:
        client.run_test(250, get_pic512_data, model_file_path, False)
