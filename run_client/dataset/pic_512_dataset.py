import os
import sys
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
            print(f"分割タイプの指定が誤っています: {str(e)}")
            sys.exit(1)
        
        except Exception as e:
            print(f"データセットの初期化に失敗しました: {str(e)}")
            sys.exit(1)
            
    def __len__(self):
        return len(self.target_img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.target_img_names[idx])
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, image


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
