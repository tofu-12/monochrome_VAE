import os
import sys
from typing import Callable
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import numpy as np
import torch
import torch.optim as optim

from model.vae import VAE, reconstruction_divergence_nexus_loss
from run.base_client import RunClient
from run.dataset.pic_512_dataset import get_pic512_data
from run.schemas import VAEHistory


class VAEClient(RunClient):
    def __init__(self):
        """
        VAEを実行するクライエントのインスタンスの初期化
        """
        super().__init__()
    
    def _training(self, batch_size: int, epoch: int, weights_file_path: str, model_file_path: str):
        """
        訓練を実行

        Args:
            batch_size: バッチサイズ
            epoch: エポック数
            weights_file_path: パラメータファイルパス
            model_file_path: モデルファイルパス
        """
        self.history = VAEHistory()
        try:
            for t in range(epoch):
                print(f"Epoch {t+1}\n-------------------------------")
                self._train_loop(batch_size)
                self._val_loop()
            print("Done!")

            # モデルの保存
            self._save_weights(weights_file_path)
            self._save_model(model_file_path)
    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Attempting to save...")
            self._save_weights(weights_file_path)
            self._save_model(model_file_path)

        except Exception as e:
            raise Exception(f"モデルの学習の途中でエラーが発生しました: {str(e)}")
            
    def _train_loop(self, batch_size: int):
        """
        訓練ループ

        Args:
            batch_size: バッチサイズ
        """
        try:
            size = len(self.dataloaders.train.dataset)
            self.model.train()

            total_epoch_loss_sum = 0
            total_processed_samples = 0


            for batch, (X, y) in enumerate(self.dataloaders.train):
                X = X.to(self.device)
                batch_size_actual = len(X)

                pred, z, z_mean, z_log_var = self.model(X)

                loss = self.loss_fn(pred, X, z_mean, z_log_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_epoch_loss_sum += loss.item()
                total_processed_samples += batch_size_actual

                current = (batch + 1) * batch_size

                if batch % 10 == 0:
                    print(f"loss: {(loss.item()/batch_size_actual):>7f}  [{current:>5d}/{size:>5d}]")

                self.history.train_loss_per_batch.append(loss.item() / batch_size_actual)

            average_epoch_loss = total_epoch_loss_sum / total_processed_samples if total_processed_samples > 0 else 0
            self.history.train_loss_per_epoch.append(average_epoch_loss)
        
        except KeyboardInterrupt:
            raise KeyboardInterrupt

        except Exception as e:
            raise Exception(f"学習ループでエラーが発生しました: {str(e)}")

    def _val_loop(self):
        """
        検証ループ
        """
        self.model.eval()

        total_val_loss_sum = 0
        all_z_batches = []

        with torch.no_grad():
            for X, y in self.dataloaders.val:
                X = X.to(self.device)

                pred, z, z_mean, z_log_var = self.model(X)
                batch_loss = self.loss_fn(pred, X, z_mean, z_log_var).item()
                total_val_loss_sum += batch_loss

                z_numpy = z.to('cpu').detach().numpy()
                all_z_batches.append(z_numpy)

        total_samples = len(self.dataloaders.val.dataset)
        average_val_loss = total_val_loss_sum / total_samples if total_samples > 0 else 0

        if all_z_batches:
            val_z_concatenated = np.concatenate(all_z_batches, axis=0)
        else:
            val_z_concatenated = np.array([])

        print(f"Test Error: \n Avg loss: {average_val_loss:>8f} \n")

        self.history.val_loss_per_epoch.append(average_val_loss)
        self.history.val_z_per_epoch.append(val_z_concatenated)
    
    def set_model(self):
        """
        モデルの設定
        """
        self.device = self._get_device()
        self.model = VAE().to(self.device)

    def run_training(self, batch_size: int, epoch: int, get_data_func: Callable, weights_file_path: str, model_file_path: str, is_loading_weights: bool):
        """
        モデルの学習の実行

        Args:
            batch_size: バッチサイズ
            epoch: エポック数
            get_data_func: データ取得関数
            weights_file_path: モデルの重みを保存・読み込みするファイルのパス
            model_file_path: モデル全体を保存・読み込みするファイルのパス
            is_loading_weights: ファイルから読み込んで学習を再開するかどうかを示すフラグ
        """
        try:
            # モデルの設定
            self.set_model()

            # データの取得
            self.datasets, self.dataloaders = get_data_func(batch_size)

            # オプティマイザと損失関数の設定
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = reconstruction_divergence_nexus_loss

            # 重みのロード
            if is_loading_weights:
                self._load_params(weights_file_path)

            # モデルの学習
            self._training(batch_size, epoch, weights_file_path, model_file_path)
        
        except Exception as e:
            print(f"学習の実行の際にエラーが発生しました:\n {str(e)}")


if __name__ == "__main__":
    # 保存先のパス
    weights_file_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "model", "weights_file", "pic_512.weights.pth")
    model_file_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "model", "model_file", "pic_512.weights.pth")

    client = VAEClient()
    client.set_model()
    client.check_model_size(False)

    client.run_training(150, 5, get_pic512_data, weights_file_path, model_file_path, False)
