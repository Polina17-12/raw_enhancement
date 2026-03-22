import torch.nn as nn
from pytorch_msssim import ssim

class SSIML1Loss(nn.Module):
    def __init__(self, alpha=0.84):
        super(SSIML1Loss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        return (1.0 - self.alpha) * l1_loss + self.alpha * ssim_loss
