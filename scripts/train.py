import argparse
import torch
import os
import yaml
from torch.utils.data import DataLoader
from models.hyga_ani import HyGAANI
from datasets import get_dataset
from utils import save_model, save_config

def train(args):
    # Load configuration
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Prepare dataset and dataloaders
    train_dataset, val_dataset = get_dataset(config['dataset'], train=True, val=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = HyGAANI(config['model_params'])
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{config["epochs"]} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item()}')

        # Validation after each epoch
        validate(model, val_loader, criterion)
        
        # Save model checkpoint after each epoch
        save_model(model, optimizer, epoch, config['checkpoint_dir'])

def validate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyGA-ANI Training Script")
    parser.add_argument('--config_file', type=str, default='configs/cifar10.yaml', help='Path to config file')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args)
