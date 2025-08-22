import torch
import torch.nn.functional as F

class FeSMIAttack:
    def __init__(self, model, device, threshold=0.5):
        """
        Initializes the FeS-MIA attack with model, device, and attack threshold.

        Args:
            model (torch.nn.Module): The trained model to attack.
            device (torch.device): The device (CPU or CUDA) where model and data are.
            threshold (float): The threshold to decide if a sample was in training data or not.
        """
        self.model = model
        self.device = device
        self.threshold = threshold

    def attack(self, train_loader, test_loader):
        """
        Executes the feature-based membership inference attack.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.

        Returns:
            accuracy (float): The success rate of the attack.
        """
        self.model.eval()

        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Get the model's output (either raw logits or probabilities)
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)

            # Using the maximum probability as a feature for membership inference
            max_confidence, _ = torch.max(probs, dim=1)
            
            # Classify based on whether the confidence exceeds the threshold
            is_member = max_confidence > self.threshold

            total += target.size(0)
            correct += (is_member == target).sum().item()

        accuracy = correct / total
        return accuracy

    def compute_fes_score(self, data_loader):
        """
        Computes the FeS-MIA attack score for a given data loader.

        Args:
            data_loader (DataLoader): The data to evaluate the attack on (e.g., train or test data).

        Returns:
            fes_score (float): The accuracy of the feature-based membership inference attack.
        """
        self.model.eval()

        fes_score = 0
        total_samples = 0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass to get the model outputs
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)

            # Extract the maximum probability (confidence) as a membership signal
            max_confidence, _ = torch.max(probs, dim=1)

            # Infer membership based on the maximum probability threshold
            is_member = max_confidence > self.threshold

            fes_score += (is_member == target).sum().item()
            total_samples += target.size(0)

        fes_score = fes_score / total_samples
        return fes_score
