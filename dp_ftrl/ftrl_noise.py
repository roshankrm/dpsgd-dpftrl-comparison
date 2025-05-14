# -----------------------------------------------------------------------------
# This code uses the privacy computation and optimizer from:
#
#   google-research/DP-FTRL  
#   https://github.com/google-research/DP-FTRL

# -----------------------------------------------------------------------------

"""The tree aggregation protocol for noise addition in DP-FTRL."""
import torch

class NoiseTreeAggregator:
    @torch.no_grad()
    def __init__(self, sigma, shapes, device):
        """
        Maintains a binary counter to reuse Gaussian noise samples O(log T)
        times instead of O(T).  Each call returns the cumulative noise
        tensor(s) to be added to gradients at this step.
    
        :param sigma: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        """
        assert sigma >= 0
        self.sigma = sigma
        self.shapes = shapes
        self.device = device
        self.step = 0
        
        # binary counter bits (LSB first)
        self.bits = [0]
        
        # cumulative noise per parameter
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        
        # noise samples stored at each tree level
        self.recorded = [[torch.zeros(shape).to(self.device) for shape in shapes]]

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        if self.sigma <= 0:
            return self.noise_sum

        self.step += 1
        level = 0
        
        # Perform binary “carry” and subtract retired noise
        while level < len(self.bits) and self.bits[level] == 1:
            self.bits[level] = 0
            for ns, re in zip(self.noise_sum, self.recorded[level]):
                ns -= re
            level += 1

        # If we carried past existing levels, extend the tree
        if level >= len(self.bits):
            self.bits.append(0)
            self.recorded.append([
                torch.zeros(shape, device=self.device) 
                for shape in self.shapes
            ])

        # Draw fresh noise at this level, add & record it
        for shape, ns, re in zip(self.shapes, self.noise_sum, self.recorded[level]):
            n = torch.normal(0, self.sigma, shape).to(self.device)
            
            ns += n
            re.copy_(n)

        self.bits[level] = 1
        return self.noise_sum
