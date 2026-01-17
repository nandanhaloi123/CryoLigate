import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # beta=1.0 is standard for SmoothL1
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)
        
        self.ligand_weight = ligand_weight
        self.ssim_weight = ssim_weight
        self.epsilon = 1e-6

    def ssim_component(self, pred, target):
        """
        Calculates SSIM based on global statistics (Contrast & Structure) 
        per 3D volume, similar to Eqs (4-6) in the paper.
        """
        # Flatten batch to (BatchSize, N_voxels)
        p_flat = pred.view(pred.size(0), -1)
        t_flat = target.view(target.size(0), -1)

        # 1. Means (Luminance)
        mu_p = p_flat.mean(dim=1, keepdim=True)
        mu_t = t_flat.mean(dim=1, keepdim=True)

        # 2. Variances (Contrast)
        var_p = ((p_flat - mu_p)**2).mean(dim=1, keepdim=True)
        var_t = ((t_flat - mu_t)**2).mean(dim=1, keepdim=True)
        
        # 3. Covariance (Structure)
        cov_pt = ((p_flat - mu_p) * (t_flat - mu_t)).mean(dim=1, keepdim=True)

        # SSIM Formula
        # (2*mu_p*mu_t + C1) * (2*cov_pt + C2)
        # ------------------------------------
        # (mu_p^2 + mu_t^2 + C1) * (var_p + var_t + C2)
        
        # Constants to prevent div/0
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        numerator = (2 * mu_p * mu_t + C1) * (2 * cov_pt + C2)
        denominator = (mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)
        
        ssim_score = numerator / denominator
        
        # Loss = 1 - SSIM (averaged over batch)
        return 1.0 - ssim_score.mean()

    def forward(self, pred, target, ligand_mask):
        # --- 1. Weighted Smooth L1 ---
        # Calculate raw loss per voxel
        raw_loss = self.smooth_l1(pred, target)
        
        # Create weight map: 1.0 for background, 500.0 for ligand
        weights = torch.ones_like(raw_loss)
        weights = weights + (ligand_mask * (self.ligand_weight - 1.0))
        
        # Apply weights
        term1_loss = (raw_loss * weights).mean()
        
        # --- 2. SSIM Loss ---
        term2_loss = 0.0
        if self.ssim_weight > 0:
            term2_loss = self.ssim_component(pred, target)
            
        return term1_loss + (self.ssim_weight * term2_loss)