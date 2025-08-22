import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class SMILEAttack:
    def __init__(self, model, device, threshold=0.7):
        """
        Initializes the SMILE attack with the model, device, and attack threshold.

        Args:
            model (torch.nn.Module): The trained model to attack.
            device (torch.device): The device (CPU or CUDA) where model and data are.
            threshold (float): Confidence threshold for membership inference.
        """
        self.model = model
        self.device = device
        self.threshold = threshold

    def attack(self, data_loader):
        """
        Executes the SMILE attack on the given data loader.

        Args:
            data_loader (DataLoader): DataLoader with test samples to attack.

        Returns:
            accuracy (float): The accuracy of the membership inference attack.
        """
        self.model.eval()

        correct = 0
        total = 0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Get the model's outputs (logits or softmax probabilities)
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)

            # Get the maximum softmax probability for each sample
            max_confidence, _ = torch.max(probs, dim=1)

            # Classify as training set if max confidence exceeds threshold
            is_member = max_confidence > self.threshold

            # Check if the attack's inference matches the true membership
            correct += (is_member == target).sum().item()
            total += target.size(0)

        accuracy = correct / total
        return accuracy

    def compute_smile_score(self, data_loader):
        """
        Computes the SMILE attack score on the given data loader.

        Args:
            data_loader (DataLoader): The data loader containing the dataset.

        Returns:
            smile_score (float): The attack success rate.
        """
        self.model.eval()

        smile_score = 0
        total_samples = 0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Get model output probabilities (softmax)
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)

            # Extract the highest softmax probability as the confidence
            max_confidence, _ = torch.max(probs, dim=1)

            # Inference: Membership if confidence is above threshold
            is_member = max_confidence > self.threshold

            smile_score += (is_member == target).sum().item()
            total_samples += target.size(0)

        smile_score = smile_score / total_samples
        return smile_score
