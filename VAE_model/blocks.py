import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int, padding: int):
        """
        畳み込みブロックのインスタンスの初期化
        Conv -> BatchNorm -> Relu

        Args:
            in_channels: 入力チャネル数
            out_channels: 出力チャネル数
            kernel_size: カーネルサイズ
            stride: ストライドの幅
            padding: パディングサイズ
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理

        Args:
            x: 入力テンソル
        
        Returns:
            torch.Tensor: 出力データ
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int, padding: int, output_padding: int):
        """
        デコーダー用の転置畳み込みブロックのインスタンスの初期化
        ConvTranspose -> BatchNorm -> Relu

        Args:
            in_channels: 入力チャネル数
            out_channels: 出力チャネル数
            kernel_size: カーネルサイズ
            stride: ストライドの幅
            padding: パディングサイズ
            output_padding: 出力の追加パディングサイズ
        """
        super(DecoderConvBlock, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の処理
        入力テンソルxに対して、ConvTranspose -> BatchNorm -> Reluの順で適用

        Args:
            x: 入力テンソル

        Returns:
            torch.Tensor: 処理後の出力テンソル
        """
        x = self.conv_t(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class Reparameterize(nn.Module):
    def __init__(self):
        """
        再パラメータ化トリックを実装するクラスのインスタンスの初期化
        """
        super(Reparameterize, self).__init__()

    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        再パラメータ化トリックにより、潜在変数zをサンプリング

        Args:
            z_mean: 潜在空間の平均を表すテンソル
            z_log_var: 潜在空間の対数分散を表すテンソル

        Returns:
            torch.Tensor: サンプリングされた潜在変数zのテンソル
        """
        std_dev = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(z_mean)
        z = z_mean + std_dev * epsilon
        
        return z    
