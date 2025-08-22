
import torch
import torch.nn.functional as F

class DynaNoise(BaseDefense):
    def __init__(self, model, device, entropy_scale=1.5, noise_level=0.1):
        super().__init__(model, device, noise_level)
        self.entropy_scale = entropy_scale

    def apply_defense(self, data):
        """
        Apply DynaNoise defense by injecting noise based on entropy.
        
        Args:
            data (torch.Tensor): The input data to apply defense on.
        
        Returns:
            torch.Tensor: The data after applying DynaNoise.
        """
        output = self.model(data)
        entropy = self.compute_entropy(output)
        noise = self.noise_level * torch.randn_like(data) * entropy
        return data + noise

    def compute_entropy(self, logits):
        """
        Compute the entropy of the model's output logits.

        Args:
            logits (torch.Tensor): The raw output logits from the model.
        
        Returns:
            torch.Tensor: The entropy values for each sample.
        """
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return torch.sigmoid(self.entropy_scale * entropy)
