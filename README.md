# monochrome_pic_autoencoder

## 概要
白黒画像を対象としたVAEモデルを実装しています。  
使用しているデータセットは以下のとおりです。

- Fashion-MNIST:  
    https://www.kaggle.com/datasets/zalando-research/fashionmnist
- pic_512:  
    オリジナルのランダム生成した線からなる白黒画像

## ファイル構造
- **app/:** モデルを実行するアプリケーション関連
- **create_data/:** pic_512を生成するプログラム関連
- **VAE_client/:** モデルを実行するクライエント関連
- **VAE_model/:** VAEのモデル関連
