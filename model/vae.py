import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2, 2)(x)
        return x


class TransposeCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(TransposeCNNBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class MonochromeEncoder(nn.Module):
    def __init__(self):
        super(MonochromeEncoder, self).__init__()
        self.cnn_block_1 = CNNBlock(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.cnn_block_2 = CNNBlock(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MonochromeDecoder(nn.Module):
    def __init__(self):
        super(MonochromeDecoder, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 16 * 7 * 7)
        self.transpose_cnn_block_1 = TransposeCNNBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.transpose_cnn_block_2 = TransposeCNNBlock(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 16, 7, 7)
        x = self.transpose_cnn_block_1(x)
        x = self.transpose_cnn_block_2(x)
        return x


class MonochromeAutoEncoder(nn.Module):
    def __init__(self):
        super(MonochromeAutoEncoder, self).__init__()
        self.encoder = MonochromeEncoder()
        self.decoder = MonochromeDecoder()
    
    def forward(self, x):
        encode = self.encoder(x)
        output = self.decoder(encode)
        return output
    