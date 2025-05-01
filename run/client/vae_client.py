import os
import sys
from typing import Callable
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import numpy as np
import matplotlib.pyplot as plt
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
        # デバイスの設定
        self.device = self._get_device()

        # モデルの設定
        self.model = VAE().to(self.device)

        # オプティマイザと損失関数の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = reconstruction_divergence_nexus_loss
    
    def set_data(self, batch_size: int, get_data_func: Callable):
        """
        データの設定

        Args:
            batch_size: バッチサイズ
            get_data_func: データ取得関数
        """
        self.datasets, self.dataloaders = get_data_func(batch_size)

    def run_training(self, batch_size: int, epoch: int, get_data_func: Callable, weights_file_path: str, model_file_path: str, loading_weights: bool):
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

            # データの設定
            self.set_data(batch_size, get_data_func)

            # オプティマイザと損失関数の設定
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = reconstruction_divergence_nexus_loss

            # 重みのロード
            if loading_weights:
                self._load_params(weights_file_path)

            # モデルの学習
            self._training(batch_size, epoch, weights_file_path, model_file_path)
        
        except Exception as e:
            print(f"学習の実行の際にエラーが発生しました:\n {str(e)}")
    
    def run_test(self, batch_size: int, get_data_func: Callable, model_file_path: str, checking_test_loss: bool):
        """
        テストの実行

        Args:
            batch_size: バッチサイズ
            get_data_func: データ取得関数
            model_file_path: モデルファイルのパス
            checking_test_loss: テストデータの損失を確認するか否か
        """
        # モデルの設定
        self.set_model()
        self._load_model(model_file_path)

        # データの設定
        self.set_data(batch_size, get_data_func)

        # モデルのモード変更
        self.model.eval()

        # テストデータの損失の確認
        if checking_test_loss:
            total_test_loss_sum = 0

            with torch.no_grad():
                for X, y in self.dataloaders.test:
                    X = X.to(self.device)

                    pred, z, z_mean, z_log_var = self.model(X)
                    batch_loss = self.loss_fn(pred, X, z_mean, z_log_var).item()
                    total_test_loss_sum += batch_loss

                total_samples = len(self.dataloaders.test.dataset)
                average_val_loss = total_test_loss_sum / total_samples if total_samples > 0 else 0
                print(average_val_loss)
        
        # 再構成の可視化
        with torch.no_grad():
            data_iter = iter(self.dataloaders.test)
            X, _ = next(data_iter)
            X = X.to(self.device)

            pred, _, _, _ = self.model(X)

            X_cpu = X.cpu().permute(0, 2, 3, 1).numpy()
            pred_cpu = pred.cpu().permute(0, 2, 3, 1).numpy()

            num_samples_to_show = min(X_cpu.shape[0], 5)

            plt.figure(figsize=(10, 4))
            for i in range(num_samples_to_show):
                # 元画像
                ax = plt.subplot(2, num_samples_to_show, i + 1)
                
                plt.imshow(X_cpu[i].squeeze(), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title('Original')

                # 再構成画像
                ax = plt.subplot(2, num_samples_to_show, i + 1 + num_samples_to_show)
                
                plt.imshow(pred_cpu[i].squeeze(), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if i == 0:
                    ax.set_title('Reconstruction')

            plt.suptitle('Original vs Reconstruction')
            plt.show()

        print("Test run complete.")


if __name__ == "__main__":
    # 保存先のパス
    weights_file_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "model", "weights_file", "pic_512.weights.pth")
    model_file_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "model", "model_file", "pic_512.weights.pth")

    is_train = True

    client = VAEClient()
    if is_train:
        client.run_training(200, 20, get_pic512_data, weights_file_path, model_file_path, False)
    else:
        client.run_test(200, get_pic512_data, model_file_path, False)
