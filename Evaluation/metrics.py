import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions.

    Args:
        predictions (torch.Tensor): The predicted labels.
        targets (torch.Tensor): The true labels.

    Returns:
        float: Accuracy value.
    """
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def precision(predictions, targets):
    """
    Calculate precision score.

    Args:
        predictions (torch.Tensor): The predicted labels.
        targets (torch.Tensor): The true labels.

    Returns:
        float: Precision value.
    """
    predicted = predictions.argmax(dim=1)
    return precision_score(targets.cpu(), predicted.cpu(), average='binary')

def recall(predictions, targets):
    """
    Calculate recall score.

    Args:
        predictions (torch.Tensor): The predicted labels.
        targets (torch.Tensor): The true labels.

    Returns:
        float: Recall value.
    """
    predicted = predictions.argmax(dim=1)
    return recall_score(targets.cpu(), predicted.cpu(), average='binary')

def f1(predictions, targets):
    """
    Calculate the F1 score.

    Args:
        predictions (torch.Tensor): The predicted labels.
        targets (torch.Tensor): The true labels.

    Returns:
        float: F1 score.
    """
    predicted = predictions.argmax(dim=1)
    return f1_score(targets.cpu(), predicted.cpu(), average='binary')

def auc(predictions, targets):
    """
    Calculate the Area Under Curve (AUC).

    Args:
        predictions (torch.Tensor): The predicted probabilities.
        targets (torch.Tensor): The true labels.

    Returns:
        float: AUC score.
    """
    predicted_probabilities = F.softmax(predictions, dim=1)[:, 1]  # Assuming binary classification
    return roc_auc_score(targets.cpu(), predicted_probabilities.cpu())
