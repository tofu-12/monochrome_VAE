import torch
import torch.nn.functional as F


def bce_reconstruction_loss(predict, target, z_mean, z_log_var):
    """
    VAEの損失関数
    BCE再構成誤差とKL情報量の和

    Args:
        predict: 予測値
        target: 真値
        z_mean: 潜在空間の平均値
        z_log_var: 潜在空間の対数分散
    
    Returns:
        損失
    """
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp())
    loss = bce_loss + kl_loss
    return loss


def weight_bce_reconstruction_loss(predict, target, z_mean, z_log_var):
    """
    VAEの損失関数
    重みつきBCE再構成誤差とKL情報量の和

    Args:
        predict: 予測値
        target: 真値
        z_mean: 潜在空間の平均値
        z_log_var: 潜在空間の対数分散
    
    Returns:
        損失
    """
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp())
    loss = bce_loss + kl_loss
    return loss


def mse_reconstruction_los(predict, target, z_mean, z_log_var):
    """
    VAEの損失関数
    MSE再構成誤差とKL情報量の和

    Args:
        predict: 予測値
        target: 真値
        z_mean: 潜在空間の平均値
        z_log_var: 潜在空間の対数分散
    
    Returns:
        損失
    """
    mse_loss = F.mse_loss(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - z_log_var.exp())
    loss = mse_loss + kl_loss
    return loss
