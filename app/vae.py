import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from utils import (
    get_device,
    get_fashion_mnist_dataset
)
from model.vae import MonochromeAutoEncoder


if __name__ == "__main__":
    # デバイスの取得
    device = get_device()
    print(f"Using device: {device}")

    # データの取得
    BATCH_SIZE = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_set, validation_set = get_fashion_mnist_dataset(transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = (
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    )

    # モデルの定義とデバイスへの転送
    model = MonochromeAutoEncoder().to(device)

    # 損失関数とオプティマイザの定義
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習ハイパーパラメータ
    NUM_EPOCHS = 10

    # 学習ループ
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # 検証ループ
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for data in validation_loader:
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                validation_loss += loss.item()

        print(f'Epoch {epoch + 1}, Validation Loss: {validation_loss / len(validation_loader):.3f}')

    print('Finished Training')

    # モデルの保存 (オプション)
    PATH = './fashion_mnist_autoencoder.pth'
    torch.save(model.state_dict(), PATH)

    # 学習済みモデルの評価 (オプション)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    # with torch.no_grad():
    #     # テストデータのロードなど
    #     pass

    # --- 復元画像の表示 ---
    print("\nDisplaying reconstruction examples...")

    # 学習済みモデルをロードする場合 (例えば別のスクリプトで実行する場合など)
    # model = MonochromeAutoEncoder(latent_dim=LATENT_DIM).to(device) # モデルインスタンスを再作成
    # model.load_state_dict(torch.load(PATH)) # 保存した重みをロード
    # print(f"Model loaded from {PATH}")

    # モデルを評価モードにする
    model.eval()

    # 検証データローダーから最初のバッチを取得
    # もし別途テストデータセットを用意しているなら、そちらを使用するのがより正確です
    dataiter = iter(validation_loader)
    images, labels = next(dataiter) # 最初のバッチを取得

    # デバイスへ転送
    images = images.to(device)

    # 画像表示のためのデノーマライズ関数
    # FashionMNISTのノーマライズは平均0.5、標準偏差0.5なので、 img * 0.5 + 0.5 で元の0-1範囲に戻せます
    def imshow_denormalize(tensor_image):
        """Helper to denormalize and convert tensor for imshow."""
        # Ensure tensor is on CPU and convert to numpy
        img = tensor_image.cpu().numpy()
        # Denormalize (reverse of: (img - 0.5) / 0.5)
        img = img * 0.5 + 0.5
        # Squeeze channel dimension for grayscale (CxHxW -> HxW)
        img = np.squeeze(img)
        # Clip values to be in [0, 1] range
        img = np.clip(img, 0, 1)
        return img


    # モデルによる画像復元 (推論時はtorch.no_grad()を使う)
    with torch.no_grad():
        reconstructed_images = model(images)

    # 最初の5枚のオリジナル画像と復元画像を比較表示
    num_images_to_show = 5
    fig, axes = plt.subplots(nrows=2, ncols=num_images_to_show, figsize=(10, 4))
    fig.suptitle(f"Original vs Reconstructed Images (First {num_images_to_show} from Validation Set)", y=1.02) # タイトルを少し上に表示

    for i in range(num_images_to_show):
        # オリジナル画像
        ax = axes[0, i]
        # デノーマライズして表示
        ax.imshow(imshow_denormalize(images[i]), cmap='gray')
        ax.set_title(f"Original\n({classes[labels[i]]})") # クラス名も表示（オプション）
        ax.axis('off') # 軸を非表示

        # 復元画像
        ax = axes[1, i]
        # モデル出力はSigmoidで0-1に正規化されているので、そのまま表示
        ax.imshow(reconstructed_images[i].cpu().squeeze().numpy(), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off') # 軸を非表示

    plt.tight_layout() # サブプロット間のスペースを調整
    plt.show() # プロットを表示
