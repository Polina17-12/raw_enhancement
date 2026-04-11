import torch.nn as nn


class ColorAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        pred_r_g = pred[:, 0] - pred[:, 1]
        pred_b_g = pred[:, 3] - pred[:, 2]
        target_r_g = target[:, 0] - target[:, 1]
        target_b_g = target[:, 3] - target[:, 2]

        loss_color = self.l1(pred_r_g, target_r_g) + self.l1(pred_b_g, target_b_g)
        return loss_l1 + 0.5 * loss_color


# Keep backward-compatible name used elsewhere in the codebase
SSIML1Loss = ColorAwareLoss
