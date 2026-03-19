import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None, df=None):
        self.data_df = pd.read_csv(csv_path)
        
        if df is not None:
            self.data_df = df

        # ✅ common transformation
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # image file path
        img_name = str(self.data_df.iloc[idx]['path'])

        # load the image
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_name}, Error: {e}")
            return torch.zeros((3, 224, 224)), -1, img_name, img_name 

        if self.transform:
            image = self.transform(image)

        # get label and score
        label = self.data_df.iloc[idx]['label']
        score = self.data_df.iloc[idx]['difficulty']
            
        # transform label to int
        try:
            label = int(label)
        except ValueError:
            print(f"Warning: Invalid label at index {idx}: {label}, setting to 0")
            label = 0 

        return image, label, score, idx


