import os
from typing import Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..schemas import Dataloaders, Datasets


class Pic512Dataset(Dataset):
    def __init__(self, img_dir: str, partition_file: str, partition_type: str, transform: Optional[callable]=None):
        """
        Pic512データセットのインスタンスの初期化

        Args:
            img_dir: 画像ファイルが格納されているディレクトリのパス
            partition_path: 分割情報 (train/val/test) が記述されたtxtファイルのパス
            partition_type: "train", "val", または "test" を指定
            transform: 画像に適用する変換処理
        """
        try:
            self.img_dir = img_dir
            self.transform = transform

            # 分割情報による使用する画像のラベルの抽出
            partition_dict = {"train": 0, "val": 1, "test": 2}
            partition_list = pd.read_csv(partition_file)

            target_partition_type = partition_dict[partition_type]
            self.target_img_names = partition_list[partition_list["type"] == target_partition_type]["file_name"].to_list()

        except KeyError as e:
            raise KeyError(f"分割タイプの指定が誤っています: {str(e)}")
        
        except Exception as e:
            raise Exception(f"データセットの初期化に失敗しました: {str(e)}")
            
    def __len__(self):
        return len(self.target_img_names)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.target_img_names[idx])
            image = Image.open(img_path).convert('L')

            if self.transform:
                image = self.transform(image)

            return image, image
        
        except Exception as e:
            raise Exception(f"アイテムの取得に失敗しました: {str(e)}")


def get_pic512_data(batch_size: int) -> tuple[Datasets, Dataloaders]:
    """
    データローダの作成

    Args:
        batch_size: バッチサイズ

    Returns:
        tuple: Datasets, Dataloaders
    """
    # パスの設定
    img_dir_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "pic_512", "image")
    partition_file_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "pic_512", "partition_file.csv")

    # データセットの作成
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = Pic512Dataset(img_dir_path, partition_file_path, "train", transform)
        val_dataset = Pic512Dataset(img_dir_path, partition_file_path, "val", transform)
        test_dataset = Pic512Dataset(img_dir_path, partition_file_path, "test", transform)
        datasets = Datasets(train=train_dataset, val=val_dataset, test=test_dataset)
    
    except Exception as e:
        raise Exception(f"データセットの作成に失敗しました: \n {str(e)}")

    # データローダの作成
    try:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)
        dataloaders = Dataloaders(train=train_dataloader, val=val_dataloader, test=test_dataloader)

        return datasets, dataloaders

    except Exception as e:
        raise Exception(f"データローダの作成に失敗しました: \n {str(e)}")
    
            

if __name__ == "__main__":
    # pathの設定
    partition_file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "pic_512", "partition_file.csv")
    img_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "pic_512", "image")

    # データセットの初期化
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = Pic512Dataset(img_dir, partition_file, "train", transform)
    print(dataset[0])
