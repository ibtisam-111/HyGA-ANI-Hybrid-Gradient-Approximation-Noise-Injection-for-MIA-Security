import torch
import torch.nn.functional as F

class MrMAttack:
    def __init__(self, model, device, threshold=0.5):
        self.model = model
        self.device = device
        self.threshold = threshold

    def attack(self, train_loader, test_loader):
        self.model.eval()

        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probs, dim=1)
            is_member = confidence > self.threshold
            
            total += target.size(0)
            correct += (is_member == target).sum().item()

        accuracy = correct / total
        return accuracy

    def compute_mrm_score(self, data_loader):
        mrm_score = 0
        total_samples = 0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            outputs = self.model(data)
            probs = F.softmax(outputs, dim=1)
            
            confidence, predicted_class = torch.max(probs, dim=1)
            is_member = confidence > self.threshold

            mrm_score += (is_member == target).sum().item()
            total_samples += target.size(0)

        mrm_score = mrm_score / total_samples
        return mrm_score
