import torch, pdb
import torch.nn as nn

class WGLSLoss(nn.Module):
   
    '''
    Weighted Generalized least squares (WGLS) loss function for proportional
    statistical error models with additional weighting based on true (target)
    values.
   
    Args:
        gamma         (float): scalar >= 0 for proportionality constant
        threshold     (float): scalar << 1 to prevent division by zero
        detach_prop    (bool): whether to detach denominator from comp. graph
        target_weight (float): weight value
        target_val    (float): target values to be weighted
   
    Inputs:
        pred (tensor): predicted variables
        true (tensor): true variables
   
    Returns:
        loss (tensor): torch float of GLS loss
    '''
   
    def __init__(self, 
                 gamma=0.0, 
                 threshold=1e-4, 
                 detach_prop=False, 
                 target_weight=1.0, 
                 target_val=None):
       
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.detach_prop = detach_prop
        self.target_weight = target_weight
        self.target_val = target_val
        
        assert self.gamma >= 0 and self.threshold >= 1e-10
       
    def forward(self, pred, true):
        
        # residual error
        residual = pred - true
        
        # if GLS (instead of OLS)
        if self.gamma > 0:
       
            # proportional weights
            residual *= pred.abs().clamp(min=self.threshold)**(-self.gamma)
            
        # if weighting
        if self.target_val is not None:
            
            # target_weight for target_val otherwise multiply by 1
            residual *= torch.where(true == self.target_val,
                                    self.target_weight*torch.ones_like(true), 
                                    torch.ones_like(true))

        # mean square proportional error
        loss = torch.mean(residual**2)
       
        return loss
