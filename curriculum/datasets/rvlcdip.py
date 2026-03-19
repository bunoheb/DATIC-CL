# The code is developed based on 
# https://github.com/THUMNLab/CurML
# Zhou, Y., Chen, H., Pan, Z., Yan, C., Lin, F., Wang, X., & Zhu, W. (2022, October). Curml: A curriculum machine learning library. In Proceedings of the 30th ACM International Conference on Multimedia (pp. 7359-7363).

import os
import pandas as pd
import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from curriculum.datasets.custom_dataset import CustomDataset
from curriculum.datasets.document_dataset import DocumentDataset
from curriculum.datasets.utils import LabelNoise

# check the none-value
def collate_fn(batch):
    batch = [b for b in batch if b is not None] 
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# common transformation
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_rvlcdip_dataset(data_path=None, noise_ratio=0.0, use_huggingface=True, batch_size=16, valid_ratio=0.2, test_ratio=0.2, shuffle=True):
    """RVL-CDIP dataset load"""
    
    print(f"🔥 use_huggingface: {use_huggingface}")

    # data path setting
    if data_path is None:
        data_path = 'data/data_with_combined_difficulty.csv'

    if use_huggingface:
        print("🚀 Using Hugging Face RVL-CDIP dataset!")
        train_dataset = DocumentDataset(split="train", transform=common_transform)
        valid_dataset = DocumentDataset(split="validation", transform=common_transform)
        test_dataset = DocumentDataset(split="test", transform=common_transform)
    else:
        print("📂 Using CSV-based dataset!")

        # csv file check
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"🚨 There is no CSV: {data_path}")

        train_dataset, valid_dataset = get_train_valid_dataset(
            csv_path=data_path, test_ratio=test_ratio, valid_ratio=valid_ratio, noise_ratio=noise_ratio, transform=common_transform
        )
        test_dataset = get_test_dataset(csv_path=data_path, test_ratio=test_ratio, transform=common_transform)
        
    print(f"✅ Final Split -> Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")  # 🚀 Split check
    return train_dataset, valid_dataset, test_dataset

def get_train_valid_dataset(csv_path, test_ratio, valid_ratio, noise_ratio, transform):
    """Load the RVL-CDIP Training and Validation set based on CSV file"""
    
    full_dataset = CustomDataset(csv_path=csv_path, transform=transform)
    
    # train / test split
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=test_ratio, random_state=42)

    # train / valida split
    train_indices, valid_indices = train_test_split(train_indices, test_size=valid_ratio, random_state=42)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)

    print(f"✅ Split complete -> Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")  


    if noise_ratio > 0.0:
        train_dataset = LabelNoise(train_dataset, noise_ratio, 16)  # 16-class 

    return train_dataset, valid_dataset

def get_test_dataset(csv_path, test_ratio, transform):
    """Load the RVL-CDIP Test set based on CSV file"""
    
    full_dataset = CustomDataset(csv_path=csv_path, transform=transform)
    
    # ✅ train / test split
    _, test_indices = train_test_split(range(len(full_dataset)), test_size=test_ratio, random_state=42)
    
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    return test_dataset

