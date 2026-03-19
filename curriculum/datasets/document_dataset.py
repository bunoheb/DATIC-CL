from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image, UnidentifiedImageError

class DocumentDataset(Dataset):
    def __init__(self, split='train', transform=None):
        valid_splits = ["train", "validation", "test"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Choose from {valid_splits}.")

        print(f"Loading Hugging Face RVL-CDIP split: {split}")
        
        # ✅ Load the data
        self.dataset = load_dataset("rvl_cdip", split=split)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.Grayscale(num_output_channels=3),  # 🚀 3channel transformation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.num_classes = 16  # Number of RVL-CDIP classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): 
        
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # 🚀 Tensor to int
        try:
            item = self.dataset[idx]
        except Exception as e:
            print(f" Error fetching item at index {idx}: {e}")
            return None 

        if not isinstance(item, dict) or "image" not in item or "label" not in item:
            print(f" Invalid data format at index {idx}: {item}")
            return None

        try:
            if not isinstance(item["image"], Image.Image):
                item["image"] = Image.fromarray(item["image"])

            image = item["image"]
            image.verify()
            image = image.convert("RGB")
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"  Skipping corrupt image at index {idx}: {e}")
            return None

        label = item["label"]
        if not isinstance(label, int):
            print(f" Warning: Invalid label type {type(label)} at index {idx}, setting to 0")
            label = 0  

        image = self.transform(image)

        return image, label, idx, f"HF_RVLCDIP_{idx}.jpg" 



# ✅ data value check
def collate_fn(batch):
    batch = [b for b in batch if b is not None]  
    
    if len(batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(batch)

def get_document_dataloaders(batch_size=16, shuffle=True):
    train_dataset = DocumentDataset('train')
    valid_dataset = DocumentDataset('validation')
    test_dataset = DocumentDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


