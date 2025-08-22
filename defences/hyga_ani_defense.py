import torch

class HyGAANI(BaseDefense):
    def __init__(self, model, device, epsilon=1e-8, alpha=0.1, beta=0.05, sigma_max=1.0):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.sigma_max = sigma_max

    def apply_defense(self, data):
        """
        Apply HyGA-ANI defense by dynamically adjusting noise based on gradient sensitivity.
        
        Args:
            data (torch.Tensor): The input data to apply defense on.
        
        Returns:
            torch.Tensor: The data after applying HyGA-ANI.
        """
        output = self.model(data)
        grad_norm = self.compute_gradient_norm(output, data)
        sigma = self.compute_noise_scale(grad_norm)
        noise = self.generate_noise(data.size(), sigma)
        return data + noise

    def compute_gradient_norm(self, output, data):
        """
        Compute the gradient norm to determine sensitivity.
        
        Args:
            output (torch.Tensor): Model output.
            data (torch.Tensor): Input data.
        
        Returns:
            float: Gradient norm value.
        """
        output.sum().backward()
        grad_norm = torch.norm(data.grad).item()
        return grad_norm

    def compute_noise_scale(self, grad_norm):
        """
        Compute the noise scale based on gradient sensitivity.
        
        Args:
            grad_norm (float): The gradient norm indicating sensitivity.
        
        Returns:
            float: Computed noise scale.
        """
        phi_x = (grad_norm - self.alpha) / (self.beta + self.epsilon)
        return min(self.alpha * phi_x ** 2 + self.beta, self.sigma_max)

    def generate_noise(self, shape, sigma):
        """
        Generate noise to inject into the data.
        
        Args:
            shape (tuple): The shape of the data.
            sigma (float): The noise scale.
        
        Returns:
            torch.Tensor: The generated noise.
        """
        return torch.normal(0, sigma, size=shape).to(self.device)

