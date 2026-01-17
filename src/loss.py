import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss to handle Class Imbalance.
    It penalizes errors inside the ligand mask more heavily than errors in the background.
    """
    def __init__(self, ligand_weight=50.0):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.ligand_weight = ligand_weight

    def forward(self, pred, target, ligand_mask):
        """
        Args:
            pred: Predicted density (Batch, 1, D, H, W)
            target: Ground truth density (Batch, 1, D, H, W)
            ligand_mask: Binary mask of ligand location (Batch, 1, D, H, W)
        """
        # 1. Calculate standard MSE per voxel
        loss = self.mse(pred, target)
        
        # 2. Create a weight map
        # Default weight = 1.0 (Background)
        # Ligand weight = ligand_weight (e.g., 50.0)
        weights = torch.ones_like(loss)
        weights = weights + (ligand_mask * (self.ligand_weight - 1.0))
        
        # 3. Apply weights and mean
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss