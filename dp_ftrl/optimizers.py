# -----------------------------------------------------------------------------
# This code uses the privacy computation and optimizer from:
#
#   google-research/DP-FTRL  
#   https://github.com/google-research/DP-FTRL

# -----------------------------------------------------------------------------


"""The DP-FTRL optimizer."""
import torch

class FTRLOptimizer(torch.optim.Optimizer):
    def __init__(self, params, momentum: float, record_last_noise: bool = True):
        """
        :param params: parameter groups
        :param momentum: if non-zero, use DP-FTRLM
        :param record_last_noise: whether to record the last noise for tree completion trick
        """
        self.momentum = momentum
        self.record_last_noise = record_last_noise
        self.current_lr = None
        self.current_noise = None
        super(FTRLOptimizer, self).__init__(params, dict())

    def __setstate__(self, state):
        super(FTRLOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Step method compatible with Opacus wrapper."""
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        # Check if params are provided as arguments or attributes
        if hasattr(self, 'current_lr') and hasattr(self, 'current_noise') and self.current_lr is not None and self.current_noise is not None:
            alpha = self.current_lr
            noise = self.current_noise
        else:
            # Default values if not set
            raise ValueError("Learning rate and noise must be set before calling step()")

        for group in self.param_groups:
            for p, nz in zip(group['params'], noise):
                if p.grad is None:
                    continue
                
                d_p = p.grad
                param_state = self.state[p]
                
                if len(param_state) == 0:
                    param_state['grad_sum'] = torch.zeros_like(d_p, memory_format=torch.preserve_format)
                    param_state['model_sum'] = p.detach().clone(memory_format=torch.preserve_format)
                    param_state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.record_last_noise:
                        param_state['last_noise'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                gs, ms = param_state['grad_sum'], param_state['model_sum']
                
                if self.momentum == 0:
                    gs.add_(d_p)
                    p.copy_(ms + (-gs - nz) / alpha)
                else:
                    gs.add_(d_p)
                    param_state['momentum'].mul_(self.momentum).add_(gs + nz)
                    p.copy_(ms - param_state['momentum'] / alpha)
                
                if self.record_last_noise:
                    param_state['last_noise'].copy_(nz)
                    
        return loss

    def set_params(self, lr, noise):
        """Set learning rate and noise for the next step."""
        self.current_lr = lr
        self.current_noise = noise
