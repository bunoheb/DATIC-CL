import torch
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer, BaseCL 

class PredefinedCurriculum(BaseCL):
    def __init__(self, schedule_type='linear', num_steps=10, epochs_per_step=3):
        """
        schedule_type: 'step', 'linear', 'root'
        num_steps: (fraction = i / num_steps)
        epochs_per_step: for step
        """
        super(PredefinedCurriculum, self).__init__()
        self.schedule_type = schedule_type
        self.num_steps = num_steps
        self.epochs_per_step = epochs_per_step
        self.epoch = 0

    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.net = net
        self.device = device
        self.criterion = criterion
        self.total_epochs = epochs


    def data_prepare(self, loader):
        self.dataloader = loader
        self.dataset = loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size

        # index alignment ascending difficulty
        self.sorted_indices = sorted(
            range(len(self.dataset)), key=lambda idx: self.dataset[idx][2]
        )

    def data_curriculum(self, loader):
        self.epoch += 1
        fraction = self._get_fraction(self.epoch)
        num_selected = max(1, int(fraction * len(self.sorted_indices)))  # choose at least 1

        selected_indices = self.sorted_indices[:num_selected]

        subset = Subset(self.dataset, selected_indices)
        return DataLoader(subset, self.batch_size, shuffle=True)

    def loss_curriculum(self, criterion, outputs, labels, indices):
        loss = criterion(outputs, labels)
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    def _get_fraction(self, epoch):
        min_fraction = 0.5  # start rate

        if self.schedule_type == 'step':
            step_index = min(epoch // self.epochs_per_step, self.num_steps - 1)
            progress = (step_index + 1) / self.num_steps
        elif self.schedule_type == 'linear':
            progress = min(epoch / self.total_epochs, 1.0)
        elif self.schedule_type == 'root':
            progress = min((epoch / self.total_epochs) ** 0.5, 1.0)
        else:
            raise NotImplementedError(f"Unknown schedule_type: {self.schedule_type}")

        # scaled progress: 0.5 ~ 1.0
        scaled_progress = min_fraction + (1.0 - min_fraction) * progress
        return scaled_progress


class PredefinedTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 batch_size, learning_rate, use_huggingface,
                 schedule_type='linear', num_steps=50, epochs_per_step=3):

        cl = PredefinedCurriculum(
            schedule_type=schedule_type,
            num_steps=num_steps,
            epochs_per_step=epochs_per_step
        )

        super(PredefinedTrainer, self).__init__(
            data_name=data_name,
            net_name=net_name,
            device_name=device_name,
            num_epochs=num_epochs,
            random_seed=random_seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_huggingface=use_huggingface,
            cl=cl
        )
