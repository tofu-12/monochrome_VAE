import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .model_run_client import ModelRunClient
from .schemas import VAEHistory


class VAERunClient(ModelRunClient):
    def __init__(self):
        """
        VAEを実行するクライエントのインスタンスの初期化
        """
        super().__init__()
    
    
    def _training(self, batch_size: int, epoch: int, weights_file_path: str, model_file_path: str) -> None:
        """
        訓練を実行

        Args:
            batch_size: バッチサイズ
            epoch: エポック数
            weights_file_path: パラメータファイルパス
            model_file_path: モデルファイルパス
        
        Raise:
            KeyboardInterrupt: Control+Cが押された場合
            Exception: その他のエラーが発生した場合
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
    
        except KeyboardInterrupt as e:
            # モデルの保存
            print("\nユーザーによる割り込みがありました")
            is_save = input("モデルを保存しますか？ (y/n) >> ")
            if is_save == "y":
                self._save_weights(weights_file_path)
                self._save_model(model_file_path)
            
            raise e

        except Exception as e:
            print(f"モデルの学習の途中でエラーが発生しました: {str(e)}")
            is_save = input("モデルを保存しますか？ (y/n) >> ")
            if is_save == "y":
                self._save_weights(weights_file_path)
                self._save_model(model_file_path)
                
            raise e

            
    def _train_loop(self, batch_size: int) -> None:
        """
        訓練ループ

        Args:
            batch_size: バッチサイズ
        
        Raise:
            KeyboardInterrupt: Control+Cが押された場合
            Exception: その他のエラーが発生した場合
        """
        try:
            size = len(self.dataloaders.train.dataset)
            self.model.train()

            total_epoch_loss_sum = 0
            total_processed_samples = 0


            for batch, (X, _) in enumerate(self.dataloaders.train):
                X = X.to(self.device)
                batch_size_actual = len(X)

                pred, _, z_mean, z_log_var = self.model(X)

                loss = self.loss_function(pred, X, z_mean, z_log_var)

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
        
        except KeyboardInterrupt as e:
            raise e

        except Exception as e:
            raise Exception(f"学習ループでエラーが発生しました: {str(e)}")


    def _val_loop(self, sample_size: int=int(1e4)) -> None:
        """
        検証ループ

        Args:
            sample_size: 検証データの個数

        Raise:
            Exception: 任意のエラーが発生した場合
        """
        try:
            self.model.eval()

            total_val_loss_sum = 0
            all_z_batches = []

            with torch.no_grad():
                sampled_val_data = self.dataloaders.val[:sample_size]
                for X, _ in sampled_val_data:
                    X = X.to(self.device)

                    pred, z, z_mean, z_log_var = self.model(X)
                    batch_loss = self.loss_function(pred, X, z_mean, z_log_var).item()
                    total_val_loss_sum += batch_loss

                    z_numpy = z.to('cpu').detach().numpy()
                    all_z_batches.append(z_numpy)

            average_val_loss = total_val_loss_sum / sample_size if sample_size > 0 else 0

            if all_z_batches:
                val_z_concatenated = np.concatenate(all_z_batches, axis=0)
            else:
                val_z_concatenated = np.array([])

            print(f"Test Error: \n Avg loss: {average_val_loss:>8f} \n")

            self.history.val_loss_per_epoch.append(average_val_loss)
            self.history.val_z_per_epoch.append(val_z_concatenated)
        
        except Exception as e:
            raise Exception(f"検証中にエラーが発生しました: {str(e)}")
        

    def run_training(
            self,
            batch_size: int,
            epoch: int,
            weights_file_name: str,
            model_file_name: str,
            loading_weights: bool=True
    ) -> None:
        """
        モデルの学習の実行

        Args:
            batch_size: バッチサイズ
            epoch: エポック数
            weights_file_path: モデルの重みを保存・読み込みするファイル名
            model_file_path: モデル全体を保存するファイル名
            is_loading_weights: ファイルからパラメータ読み込んで学習を再開するかどうかを示すフラグ
        
        Raise:
            Exception: 任意のエラーが発生した場合
        """
        try:
            # 重みのロード
            if loading_weights:
                self._load_params(weights_file_name)

            # モデルの学習
            self._training(batch_size, epoch, weights_file_name, model_file_name)
        
        except Exception as e:
            Exception(f"学習の実行の際にエラーが発生しました:\n{str(e)}")
    
    
    def run_test(
            self,
            model_file_name: str,
            checking_test_loss: bool=True
        ) -> None:
        """
        テストの実行

        Args:
            model_file_path: モデルファイル名
            checking_test_loss: テストデータの損失を確認するか否か
        
        Raise:
            Exception: 任意のエラーが発生した場合
        """
        try:
            # モデルの設定
            self._load_model(model_file_name)

            # モデルのモード変更
            self.model.eval()

            # テストデータの損失の確認
            if checking_test_loss:
                total_test_loss_sum = 0

                with torch.no_grad():
                    for X, _ in self.dataloaders.test:
                        X = X.to(self.device)

                        pred, _, z_mean, z_log_var = self.model(X)

                        batch_loss = self.loss_function(pred, X, z_mean, z_log_var).item()
                        total_test_loss_sum += batch_loss

                    total_samples = len(self.dataloaders.test.dataset)
                    average_val_loss = total_test_loss_sum / total_samples if total_samples > 0 else 0
                    print(f"Average Test Loss: {average_val_loss}")

            # 再構成の可視化
            with torch.no_grad():
                data_iter = iter(self.dataloaders.test)
                X, _ = next(data_iter)
                X = X.to(self.device)

                # predを取得
                pred, _, _, _ = self.model(X)

                # sigmoidに通す
                pred = F.sigmoid(pred)

                X_cpu = X.cpu().permute(0, 2, 3, 1).numpy()
                pred_cpu = pred.cpu().permute(0, 2, 3, 1).numpy()

                num_samples_to_show = min(X_cpu.shape[0], 5)

                plt.figure(figsize=(10, 4))
                for i in range(num_samples_to_show):
                    # 元画像の表示
                    ax = plt.subplot(2, num_samples_to_show, i + 1)

                    plt.imshow(X_cpu[i].squeeze(), cmap='gray')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    if i == 0:
                        ax.set_title('Original')

                    # 再構成画像の表示
                    ax = plt.subplot(2, num_samples_to_show, i + 1 + num_samples_to_show)
                    plt.imshow(pred_cpu[i].squeeze(), cmap="gray")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    if i == 0:
                        ax.set_title("Reconstruction")

                plt.suptitle("Original vs Reconstruction")
                plt.show()

            print("Test run complete.")
        
        except Exception as e:
            Exception(f"テストの際にエラーが発生しました:\n{str(e)}")
    

    def visualize_final_z(self) -> None:
        """
        学習履歴（損失）を可視化する

        Raise:
            Exception: 任意のエラーが発生した場合
        """
        try:
            if not hasattr(self, 'history') or self.history is None:
                print("学習履歴がありません。モデルを訓練してから実行してください。")
                return

            history = self.history

            # 最後のepochのzの分布の可視化
            if history.val_z_per_epoch:
                print("Plotting latent space distribution for the last epoch...")
                last_epoch_z = history.val_z_per_epoch[-1]
                if last_epoch_z.shape[1] >= 2:
                    plt.figure(figsize=(8, 8))
                    plt.scatter(last_epoch_z[:, 0], last_epoch_z[:, 1], s=1)
                    plt.title('Latent Space Distribution (Last Epoch)')
                    plt.xlabel('Dimension 1')
                    plt.ylabel('Dimension 2')
                    plt.grid(True)
                    plt.show()
                else:
                    print("Latent space dimension is less than 2, skipping latent space 2D plot.")

        except Exception as e:
            raise Exception(f"最後の潜在空間の可視化に失敗しました: {str(e)}")
