import os

import torch


class RunClient:
    def __init__(self):
        """
        モデルを実行するクライエントのインスタンスの初期化
        """
        self.device = None

        self.datasets = None
        self.dataloaders = None

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        self.history = None
    
    def _get_device(self) -> torch.device:
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
    
    def _train_loop(self):
        raise NotImplementedError
    
    def _val_loop(self):
        raise NotImplementedError
    
    def _training(self):
        raise NotImplementedError
    
    def _save_weights(self, weights_file_path: str) -> None:
        """
        モデルのパラメータを保存する

        Args:
            weights_file_path: パラメータファイルパス

        Raise:
            SaveModelError
        """
        try:
            print("-" * 50)
            print("Saving Parameter")
            torch.save(self.model.state_dict(), weights_file_path)
            print("Done!")
            print("-" * 50)
        
        except Exception as e:
            raise Exception(f"パラメータの保存に失敗しました: {str(e)}")
    
    def _save_model(self, model_file_path: str) -> None:
        """
        モデルを保存する

        Args:
            model_file_path: モデルファイルパス

        Raise:
            SaveModelError
        """
        try:
            print("-" * 50)
            print("Saving Model")
            torch.save(self.model, model_file_path)
            print("Done!")
            print("-" * 50)
        
        except Exception as e:
            raise Exception(f"モデルの保存に失敗しました: {str(e)}")
    
    def _load_params(self, weights_file_path: str) -> None:
        """
        モデルにパラメータをロードする

        Args:
            model: ロードするモデル
            weights_file_path: パラメータファイルパス
        
        Raise:
            LoadModelError
        """
        try:
            if os.path.exists(weights_file_path):
                print("-" * 50)
                print("Load Parameter")
                self.model.load_state_dict(torch.load(weights_file_path))
                print("Done!")
                print("-" * 50)
        
        except Exception as e:
            print(f"パラメータのロードに失敗しました: {str(e)}")
        
    def _load_model(self, model_file_path: str) -> None:
        """
        モデルをロードする

        Args:
            model_file_path: モデルファイルパス
        
        Raise:
            LoadModelError
        """
        try:
            if os.path.exists(model_file_path):
                print("-" * 50)
                print("Load Model")
                self.model = torch.load(model_file_path, weights_only=False)
                print("Done!")
                print("-" * 50)
        
        except Exception as e:
            raise Exception(f"モデルのロードに失敗しました: {str(e)}")
    
    def check_model_size(self, checking_model_structure: bool):
        """
        モデルのサイズを確認

        Args:
            is_checking_model_structure: モデルの構造を確認するか否か
        """
        if self.model:
            # トータルパラメータ数
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"モデル全体のパラメータ数: {total_params}")

            # 学習可能パラメータ数
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"学習対象のパラメータ数: {trainable_params}")

            # モデルの構造
            if checking_model_structure:
                for name, parameter in self.model.named_parameters():
                    print(f"Layer: {name} | Size: {parameter.size()} | Numel: {parameter.numel()} | Requires_grad: {parameter.requires_grad}")

        else:
            print("モデルが設定されていません")
    
    def set_model(self):
        raise NotImplementedError
    
    def set_data(self):
        raise NotImplementedError
    
    def run_training(self):
        raise NotImplementedError
    
    def run_test(self):
        raise NotImplementedError

    def visualize_history(self):
        raise NotImplementedError
