import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import torchvision
from torchvision import transforms
import scipy.io as sio

class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        
        # Load CIFAR-10 data
        if self.train:
            data_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            data_files = ['test_batch']
            
        self.data = []
        self.labels = []
        
        for file_name in data_files:
            with open(os.path.join(data_path, file_name), 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                self.data.append(batch['data'])
                self.labels.extend(batch['labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SVHNDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        
        # Load SVHN data (assuming .mat format)
        if self.train:
            data = sio.loadmat(os.path.join(data_path, 'train_32x32.mat'))
        else:
            data = sio.loadmat(os.path.join(data_path, 'test_32x32.mat'))
            
        self.data = data['X'].transpose(3, 0, 1, 2)  # Convert to NHWC
        self.labels = data['y'].flatten()
        self.labels[self.labels == 10] = 0  # SVHN has 0-9 labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PurchaseDataset(Dataset):
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.train = train
        
        # Load Purchase-100 dataset
        if self.train:
            features = np.load(os.path.join(data_path, 'purchase_train_features.npy'))
            labels = np.load(os.path.join(data_path, 'purchase_train_labels.npy'))
        else:
            features = np.load(os.path.join(data_path, 'purchase_test_features.npy'))
            labels = np.load(os.path.join(data_path, 'purchase_test_labels.npy'))
            
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TexasDataset(Dataset):
    def __init__(self, data_path, train=True):
        self.data_path = data_path
        self.train = train
        
        # Load Texas-100 dataset
        if self.train:
            features = np.load(os.path.join(data_path, 'texas_train_features.npy'))
            labels = np.load(os.path.join(data_path, 'texas_train_labels.npy'))
        else:
            features = np.load(os.path.join(data_path, 'texas_test_features.npy'))
            labels = np.load(os.path.join(data_path, 'texas_test_labels.npy'))
            
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataset(dataset_name, batch_size=128, train=True):
    """Get dataset and dataloader for the specified dataset"""
    
    data_paths = {
        'cifar10': './datasets/cifar-10-batches-py',
        'svhn': './datasets/svhn',
        'purchase100': './datasets/purchase100',
        'texas100': './datasets/texas100'
    }
    
    data_path = data_paths[dataset_name]
    
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = CIFAR10Dataset(data_path, train=train, transform=transform)
        
    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        dataset = SVHNDataset(data_path, train=train, transform=transform)
        
    elif dataset_name == 'purchase100':
        dataset = PurchaseDataset(data_path, train=train)
        
    elif dataset_name == 'texas100':
        dataset = TexasDataset(data_path, train=train)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,
        num_workers=4 if train else 2,
        pin_memory=True
    )
    
    return dataloader
