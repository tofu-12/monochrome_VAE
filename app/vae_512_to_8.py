import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from run_client.client.vae_client import VAEClient
from model.VAE.vae_512_to_8 import VAE, reconstruction_mse_loss_function, reconstruction_bce_loss_function
from run_client.dataset.pic_512_dataset import get_pic512_data


if __name__ == "__main__":
    # 保存先のパス
    weights_file_path = os.path.join(os.path.dirname(__file__), os.pardir, "model", "weights_file", "pic_512_to_8_weights.pth")
    model_file_path = os.path.join(os.path.dirname(__file__), os.pardir, "model", "model_file", "pic_512_to_8_model.pth")

    # モードの設定
    print("1: train")
    print("2: test")
    is_train = input("Select mode: ")

    if is_train == "1":
        is_train = True
    else:
        is_train = False

    # モデルの使用
    client = VAEClient(VAE)
    if is_train:
        client.run_training(250, 3, get_pic512_data, reconstruction_bce_loss_function, weights_file_path, model_file_path, True)
    else:
        client.run_test(250, get_pic512_data, reconstruction_bce_loss_function, model_file_path, True)
