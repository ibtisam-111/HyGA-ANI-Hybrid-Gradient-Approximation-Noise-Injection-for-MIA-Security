import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from models.hyga_ani import HyGAANI
from datasets import get_dataset
from evaluation.metrics import evaluate_metrics
from evaluation.visualize import plot_confusion_matrix

def evaluate(args):
    # Load configuration
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Prepare dataset and dataloaders
    _, test_dataset = get_dataset(config['dataset'], train=False, val=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = HyGAANI(config['model_params'])
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Evaluate model
    evaluate_metrics(model, test_loader)

    # Save confusion matrix
    plot_confusion_matrix(model, test_loader, args.confusion_matrix_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyGA-ANI Evaluation Script")
    parser.add_argument('--config_file', type=str, default='configs/cifar10.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--confusion_matrix_path', type=str, default='results/cifar10/confusion_matrix.png', help='Path to save confusion matrix plot')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args)
