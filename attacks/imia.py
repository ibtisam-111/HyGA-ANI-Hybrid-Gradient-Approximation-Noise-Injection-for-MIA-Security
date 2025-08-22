import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

class IMIAttack:
    def __init__(self, model, device, threshold=0.7):
        """
        Initializes the IMIA attack with the model, device, and attack threshold.

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
        Executes the IMIA attack on the given data loader.

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

            # Get the model's logits (or softmax probabilities)
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

    def compute_imia_score(self, data_loader):
        """
        Computes the IMIA attack score on the given data loader.

        Args:
            data_loader (DataLoader): The data loader containing the dataset.

        Returns:
            imia_score (float): The attack success rate.
        """
        self.model.eval()

        imia_score = 0
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

            imia_score += (is_member == target).sum().item()
            total_samples += target.size(0)

        imia_score = imia_score / total_samples
        return imia_score

    def targeted_attack(self, data_loader, target_class):
        """
        Targeted variant of the IMIA attack that attempts to infer membership of a specific class.

        Args:
            data_loader (DataLoader): DataLoader with test samples to attack.
            target_class (int): The target class for membership inference (e.g., 0 for class 0).

        Returns:
            accuracy (float): The accuracy of the targeted membership inference attack.
        """
        self.model.eval()

        correct = 0
        total = 0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Get the model's logits (or softmax probabilities)
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)

            # Get the maximum softmax probability for each sample
            max_confidence, _ = torch.max(probs, dim=1)

            # Check if the sample belongs to the target class
            is_target_class = target == target_class

            # Classify as member if max confidence exceeds threshold
            is_member = max_confidence > self.threshold

            # Check if the attack’s inference matches the true membership
            correct += ((is_member == is_target_class) & (is_target_class == target)).sum().item()
            total += target.size(0)

        accuracy = correct / total
        return accuracy

