
import torch

class AdaMixup(BaseDefense):
    def __init__(self, model, device, alpha=0.4, noise_level=0.1):
        super().__init__(model, device, noise_level)
        self.alpha = alpha

    def apply_defense(self, data):
        """
        Apply AdaMixup by mixing samples dynamically.
        
        Args:
            data (torch.Tensor): The input data to apply defense on.
        
        Returns:
            torch.Tensor: The mixed data after applying AdaMixup.
        """
        mixed_data = self.mix_samples(data)
        return mixed_data

    def mix_samples(self, data):
        """
        Perform mixing between samples based on an interpolation factor.
        
        Args:
            data (torch.Tensor): The input data to mix.
        
        Returns:
            torch.Tensor: The mixed data.
        """
        batch_size = data.size(0)
        indices = torch.randperm(batch_size).to(self.device)
        lambda_ = torch.distributions.Beta(self.alpha, self.alpha).sample((1,)).item()
        mixed_data = lambda_ * data + (1 - lambda_) * data[indices]
        return mixed_data
