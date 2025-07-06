import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class MaskedCrossEntropyLoss(nn.Module):
    """
    Masked cross entropy loss for segmentation task.
    """
    def __init__(self, loss_weight=1.0, loss_name='loss_masked_ce'):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
    
    def forward(self, input, target, mask):
        """
        Args:
            input: (N, C, H, W) - predictions from the model
            target: (N, H, W) - ground truth labels
            mask: (N, H, W) - binary mask indicating the foreground (1 for foreground, 0 for background)
        Returns:
            loss: masked cross entropy loss
        """
        # Ensure the input and target have the same number of elements as the mask
        assert input.size(0) == target.size(0) == mask.size(0)
        assert input.size(2) == target.size(1) == mask.size(1)
        assert input.size(3) == target.size(2) == mask.size(2)

        # Flatten the input, target, and mask tensors
        input = input.permute(0, 2, 3, 1).reshape(-1, input.size(1))  # (N*H*W, C)
        target = target.view(-1)  # (N*H*W)
        mask = mask.view(-1)  # (N*H*W)

        # Compute the cross entropy loss for each element
        loss = F.cross_entropy(input, target, reduction='none')  # (N*H*W)

        # Apply the mask
        loss = loss * mask

        # Compute the average loss over the foreground elements
        loss = loss.sum() / mask.sum()

        return loss*self.loss_weight
    
    @property
    def loss_name(self):
        return self._loss_name