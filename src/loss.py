import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. NEW STRATEGY: Hybrid ROI Loss (Dice + Masked MSE + Global MSE + Optional SSIM)
# ============================================================================
class HybridROILoss(nn.Module):
    """
    The "Fundamental Change" Loss.
    
    Focuses explicitly on:
    1. Shape Overlap (Dice) - Forces the model to predict a blob, not dust.
    2. Ligand Accuracy (Masked MSE) - Forces the density INSIDE the ligand to be perfect.
    3. Background Cleanup (Global MSE) - Keeps the zeros zero.
    4. Structure (SSIM) - Optional, for texture matching.
    """
    def __init__(self, dice_weight=0.5, masked_weight=10.0, global_weight=0.1, ssim_weight=0.0):
        super(HybridROILoss, self).__init__()
        self.dice_weight = dice_weight
        self.masked_weight = masked_weight
        self.global_weight = global_weight
        self.ssim_weight = ssim_weight
        
        self.mse = nn.MSELoss()
        self.smooth = 1e-5

    def _ssim_component(self, pred, target):
        """
        Calculates SSIM based on global statistics (Contrast & Structure).
        Reused from original EMReady implementation.
        """
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)

        mu_p = p_flat.mean(dim=1, keepdim=True)
        mu_t = t_flat.mean(dim=1, keepdim=True)
        var_p = ((p_flat - mu_p)**2).mean(dim=1, keepdim=True)
        var_t = ((t_flat - mu_t)**2).mean(dim=1, keepdim=True)
        cov_pt = ((p_flat - mu_p) * (t_flat - mu_t)).mean(dim=1, keepdim=True)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        numerator = (2 * mu_p * mu_t + C1) * (2 * cov_pt + C2)
        denominator = (mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)
        
        ssim_score = numerator / denominator
        return 1.0 - ssim_score.mean()

    def forward(self, pred, target, mask):
        # --- 1. Global MSE (Background Cleanup) ---
        global_loss = self.mse(pred, target)

        # --- 2. Masked MSE (Ligand Accuracy) ---
        # We add epsilon to denominator to prevent NaN if mask is empty
        mask_sum = mask.sum() + 1e-6
        masked_error = (pred - target) ** 2
        # Only sum errors where mask == 1
        masked_loss = (masked_error * mask).sum() / mask_sum

        # --- 3. Dice Loss (Shape / Overlap) ---
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # We treat the density as a "soft probability" for Dice
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth)
        dice_loss = 1 - dice_score

        # --- 4. Optional SSIM ---
        ssim_loss = torch.tensor(0.0, device=pred.device)
        if self.ssim_weight > 0:
            ssim_loss = self._ssim_component(pred, target)

        # --- Combine ---
        total_loss = (self.dice_weight * dice_loss) + \
                     (self.masked_weight * masked_loss) + \
                     (self.global_weight * global_loss) + \
                     (self.ssim_weight * ssim_loss)
                     
        return total_loss, dice_loss, masked_loss, global_loss, ssim_loss


# ============================================================================
# 2. ORIGINAL STRATEGY: EMReady (Weighted Smooth L1 + SSIM)
# ============================================================================
class EMReadyLikeLoss(nn.Module):
    """
    Implementation of the loss strategy from the EMReady paper, adapted for Ligand Imbalance.
    
    Components:
    1. Weighted Smooth L1 Loss: 
       - Acts like MSE for small errors (smooth gradients).
       - Acts like L1 for large errors (robust to outliers).
       - WEIGHED 500x inside the ligand mask to prevent empty predictions.
       
    2. SSIM Loss (Global):
       - Matches the Contrast (Standard Deviation) and Structure (Covariance).
    """
    def __init__(self, ligand_weight=500.0, ssim_weight=0.2):
        super(EMReadyLikeLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)
        self.ligand_weight = ligand_weight
        self.ssim_weight = ssim_weight

    def ssim_component(self, pred, target):
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)

        mu_p = p_flat.mean(dim=1, keepdim=True)
        mu_t = t_flat.mean(dim=1, keepdim=True)

        var_p = ((p_flat - mu_p)**2).mean(dim=1, keepdim=True)
        var_t = ((t_flat - mu_t)**2).mean(dim=1, keepdim=True)
        
        cov_pt = ((p_flat - mu_p) * (t_flat - mu_t)).mean(dim=1, keepdim=True)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        numerator = (2 * mu_p * mu_t + C1) * (2 * cov_pt + C2)
        denominator = (mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)
        
        ssim_score = numerator / denominator
        return 1.0 - ssim_score.mean()

    def forward(self, pred, target, ligand_mask):
        # --- 1. Weighted Smooth L1 ---
        raw_loss = self.smooth_l1(pred, target)
        
        # Create weight map: 1.0 for background, ligand_weight for ligand
        weights = torch.ones_like(raw_loss)
        weights = weights + (ligand_mask * (self.ligand_weight - 1.0))
        
        term1_loss = (raw_loss * weights).mean()
        
        # --- 2. SSIM Loss ---
        term2_loss = 0.0
        if self.ssim_weight > 0:
            term2_loss = self.ssim_component(pred, target)
            
        return term1_loss + (self.ssim_weight * term2_loss)