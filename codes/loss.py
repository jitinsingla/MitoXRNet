import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(RobustDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        """
        preds: shape (N, 2, H, W) – raw logits (use sigmoid later)
        targets: shape (N, 2, H, W) – binary masks
        """
        preds = torch.sigmoid(preds)  # Apply sigmoid per channel

        intersection = torch.sum(preds * targets, dim=(0, 2, 3,4))
        preds_sq = torch.sum(preds ** 2, dim=(0, 2, 3,4))
        targets_sq = torch.sum(targets ** 2, dim=(0, 2, 3,4))

        dice = (2. * intersection + self.epsilon) / (preds_sq + targets_sq + self.epsilon)
        dice_loss = 1. - dice.mean()
        return dice_loss
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()  # Works with raw logits
        self.dice = RobustDiceLoss()

    def forward(self, preds, targets):
        """
        preds: (N, 2, H, W) – raw logits
        targets: (N, 2, H, W) – binary masks
        """
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.alpha * bce_loss + self.beta * dice_loss
