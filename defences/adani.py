import torch

class AdaNI(BaseDefense):
    def __init__(self, model, device, correlation_threshold=0.5, noise_level=0.1):
        super().__init__(model, device, noise_level)
        self.correlation_threshold = correlation_threshold

    def apply_defense(self, data):
        """
        Apply AdaNI defense by injecting noise based on feature correlations.
        
        Args:
            data (torch.Tensor): The input data to apply defense on.
        
        Returns:
            torch.Tensor: The data after applying AdaNI.
        """
        correlations = self.compute_feature_correlation(data)
        noise = self.noise_level * torch.randn_like(data)
        data = data + (noise * correlations).unsqueeze(1).expand_as(data)
        return data

    def compute_feature_correlation(self, data):
        """
        Compute the correlation between features of the input data.
        
        Args:
            data (torch.Tensor): The input data to compute correlations on.
        
        Returns:
            torch.Tensor: Correlation values for each sample.
        """
        correlation_matrix = torch.corrcoef(data.T)
        return (correlation_matrix > self.correlation_threshold).float()

