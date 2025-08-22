import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HyGAANI(nn.Module):
    def __init__(self, model, alpha=0.1, beta=0.05, sigma_max=1.0, epsilon=1e-8):
        super(HyGAANI, self).__init__()
        self.model = model
        
        self.alpha = alpha
        self.beta = beta
        self.sigma_max = sigma_max
        self.epsilon = epsilon

    def forward(self, x):
        return self.model(x)

    def calculate_gradient_norm(self, loss, params):
        grad_norms = []
        for param in params:
            if param.grad is not None:
                grad_norms.append(param.grad.data.norm(2).item())
        return grad_norms

    def calculate_vulnerability_score(self, x, y, loss, step):
        self.model.zero_grad()
        loss.backward()
        grad_norms = self.calculate_gradient_norm(loss, self.model.parameters())
        grad_norm = np.mean(grad_norms)
        
        if not hasattr(self, 'mu_g'):
            self.mu_g = grad_norm
            self.sigma_g = 0
        else:
            self.mu_g = 0.9 * self.mu_g + 0.1 * grad_norm
            self.sigma_g = 0.9 * self.sigma_g + 0.1 * (grad_norm - self.mu_g) ** 2
        
        vulnerability_score = (grad_norm - self.mu_g) / (self.sigma_g + self.epsilon)
        return vulnerability_score

    def apply_noise(self, vulnerability_score, weight, batch_size):
        sigma = min(self.alpha * vulnerability_score ** 2 + self.beta, self.sigma_max)
        noise = torch.normal(mean=torch.zeros_like(weight), std=sigma * torch.ones_like(weight))
        weight.data.add_(noise)

    def apply_gradient_noise(self, x, y, loss, step):
        vulnerability_score = self.calculate_vulnerability_score(x, y, loss, step)
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.apply_noise(vulnerability_score, param, x.size(0))

    def train_step(self, x, y, optimizer, step):
        self.model.train()
        optimizer.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.apply_gradient_noise(x, y, loss, step)
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(self, x, y):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            preds = torch.argmax(output, dim=1)
            accuracy = (preds == y).float().mean().item()

        return loss.item(), accuracy
