import torch

class NoiseTransition(BaseDefense):
    def __init__(self, model, device, noise_matrix=None, noise_level=0.1):
        super().__init__(model, device, noise_level)
        self.noise_matrix = noise_matrix if noise_matrix is not None else torch.eye(10)  # Default 10 classes

    def apply_defense(self, data):
        """
        Apply Noise Transition by adding noise based on the transition matrix.
        
        Args:
            data (torch.Tensor): The input data to apply defense on.
        
        Returns:
            torch.Tensor: The data after applying Noise Transition.
        """
        output = self.model(data)
        noise = self.noise_matrix[torch.argmax(output, dim=1)] * self.noise_level
        return data + noise.unsqueeze(1).expand_as(data)

