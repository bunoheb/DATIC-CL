import torch
from torch.utils.data import Dataset, DataLoader

from ..trainers import *



class BaseCL():
    """The base class of CL Algorithm class.

    Each CL Algorithm class has a CLDataset and five key APIs.

    Attributes:
        name: A string for the name of a CL algorithm.
        dataset: A CLDataset build by the original training dataset.
        data_size: An integer for the number of training data samples.
        batch_size: An integer for the number of a mini-batch.
        n_batches: An integer for the number of batches.
    """

    class CLDataset(Dataset):
        """A dataset for CL Algorithm.

        It attaches the original training dataset with data index,
        which is a common strategy for data sampling or reweighting.
        """
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            data = self.dataset[index]
            return [part for part in data] + [index]    # Attach data index.

        def __len__(self):
            return len(self.dataset)


    def __init__(self):
        self.name = 'base'


    def data_prepare(self, loader):
        """Pass training data information from Model Trainer to CL Algorithm.
        
        Initiate the CLDataset and record training data attributes.
        """
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1


    def model_prepare(self, net, device, epochs, 
                      criterion, optimizer, lr_scheduler):
        """Pass model information from Model Trainer to CL Algorithm."""
        pass


    def data_curriculum(self, loader):
        """Measure data difficulty and schedule the training set."""
        return DataLoader(self.dataset, self.batch_size, shuffle=True)


    def model_curriculum(self, net):
        """Schedule the model changing."""
        return net


    def loss_curriculum(self, criterion, outputs, labels, indices):
        """Reweight loss."""
        return torch.mean(criterion(outputs, labels))


import numpy as np
import random
import torch.nn as nn
import torch.optim as optim

class BaseTrainer():
    """The base class of CL Trainer class.

    It initiates the Model Trainer class and CL Algorithm class, 
    and provides functions for training and evaluation.

    Attributes:
        trainer: A image classifier, language model, recommendation system, etc.
    """

    def __init__(self, data_name, net_name, device_name, 
                 num_epochs, random_seed, batch_size, learning_rate, use_huggingface, cl=BaseCL()):
        """Initiate the Model Trainer according to data_name."""
        
        if data_name.startswith('rvl') or data_name.startswith('imagenet'):
            self.trainer = ImageClassifier(
                data_name, net_name, device_name, num_epochs, random_seed, 
                cl.name, cl.data_prepare, cl.model_prepare,
                cl.data_curriculum, cl.model_curriculum, cl.loss_curriculum, 
                batch_size, learning_rate, use_huggingface=use_huggingface
            )
        else:
            raise NotImplementedError()

    def fit(self):
        return self.trainer.fit()

    def evaluate(self, net_dir=None):
        """Evaluate the net performance if given its path, else evaluate the trained net."""
        return self.trainer.evaluate(net_dir)

    def export(self, net_dir=None):
        """Load the net state dict if given its path, else load the trained net."""
        return self.trainer.export(net_dir)

    def test(self, test_loader, device):
        """Evaluate the model on the test dataset."""
        model = self.trainer.net
        model.to(device)
        model.eval()

        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels, *_ = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        return test_loss, accuracy

    def initialize_model(self, reset_model=False, seed=None):
        """Initialize the classifier model"""

        # ✅ If training phase, use `args.seed` 
        if seed is None:
            seed = args.seed  

        if reset_model:
            # Seed setting when initializing classifier
            fixed_seed = seed if seed is not None else random.randint(0, 1000000)
            random.seed(fixed_seed)
            torch.manual_seed(fixed_seed)
            torch.cuda.manual_seed_all(fixed_seed)

            print(f"🔄 New seed: {fixed_seed}")
            print("🔄 Initialize the model weight")

            self.trainer.net.apply(self.weights_init)  # weight re-initilization

            # ✅ Optimizer and Scheduler
            self.trainer.optimizer = torch.optim.Adam(self.trainer.net.parameters(), lr=self.trainer.learning_rate)
            self.trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.trainer.optimizer, T_max=self.trainer.epochs, eta_min=1e-6
            )
        else:
            print("✅ keep the model weight.")

        print("✅ Model initilization complete." if reset_model else "✅ keep the model weight.")
        
    @staticmethod
    def weights_init(m):
        """random initialize the model weight"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
