import os
import time
import torch

from ..datasets import get_dataset_with_noise
from ..backbones import get_net
from ..utils import get_logger, set_random

class ImageClassifier():
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 algorithm_name, data_prepare, model_prepare, data_curriculum, 
                 model_curriculum, loss_curriculum, batch_size, learning_rate=1e-4, 
                 use_huggingface=False): 
        self.random_seed = random_seed
        
        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum
        self.batch_size = batch_size  
        self.learning_rate = learning_rate
        self.use_huggingface = use_huggingface  

        self._init_dataloader(data_name)
        self._init_model(data_name, net_name, device_name, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)

    def _init_dataloader(self, data_name):
        print(f"🔥 use_huggingface in ImageClassifier: {self.use_huggingface}")  # for debugging

        # ✅ set the dynamic data path
        if not self.use_huggingface:
            data_path = 'data/data_with_combined_difficulty.csv'  # ✅ CSV path
        else:
            data_path = None  

        print(f"🔥 Data path in ImageClassifier: {data_path}")  # or debugging

        # ✅ Using `data_path`
        train_dataset, valid_dataset, test_dataset = get_dataset_with_noise(
            data_path, data_name, use_huggingface=self.use_huggingface
        )

        # ✅ `collate_fn`
        if self.use_huggingface:
            from ..datasets.document_dataset import collate_fn
        else:
            collate_fn = None 

        # ✅ DataLoader generation
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn
        )

        # data prepare function
        self.data_prepare(self.train_loader)


    def _init_model(self, data_name, net_name, device_name, num_epochs):
        self.net = get_net(net_name, data_name)
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

        self.model_prepare(self.net, self.device, self.epochs, 
                           self.criterion, self.optimizer, self.lr_scheduler)

    def _init_logger(self, algorithm_name, data_name, 
                     net_name, num_epochs, random_seed):
        log_info = '%s-%s-%s-%d-%d-%s' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        )
        self.log_dir = os.path.join('./temps', log_info)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_interval = 1
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), log_info)
        return

    def _train(self):
        best_loss = float('inf')

        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            correct = 0
            train_loss = 0.0

            loader = self.data_curriculum(self.train_loader)  
            net = self.model_curriculum(self.net)            

            net.train()
            for step, data in enumerate(loader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                indices = data[2] if len(data) > 2 else None  
                image_paths = data[3] if len(data) > 3 else None  

                self.optimizer.zero_grad()
                loss = self.loss_curriculum(self.criterion, net(inputs), labels, indices)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = net(inputs).max(dim=1)
                correct += predicted.eq(labels).sum().item()
                total += labels.shape[0]

            self.lr_scheduler.step()
            self.logger.info(
                '[%3d]  Train data = %6d  Train Acc = %.4f  Loss = %.4f  Time = %.2f'
                % (epoch + 1, total, correct / total, train_loss / (step + 1), time.time() - t)
            )

            if (epoch + 1) % self.log_interval == 0:
                val_loss, valid_acc = self._valid(self.valid_loader)  
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid data = %6d  Valid Acc = %.4f  Loss = %.4f'
                    % (epoch + 1, len(self.valid_loader.dataset), valid_acc, val_loss)
                )

        torch.save(self.net.state_dict(), os.path.join(self.log_dir, 'last_net.pkl'))

    def _valid(self, loader):
        total = 0
        correct = 0
        val_loss = 0.0  

        self.net.eval()
        with torch.no_grad():
            for data in loader:
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels).mean()
                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()
        return val_loss / len(loader), correct / total  

    def fit(self):
        set_random(self.random_seed)
        self._train()

    def evaluate(self, net_dir=None, use_last_model=False):
        if not use_last_model:
            self._load_best_net(net_dir)

        valid_loss, valid_acc = self._valid(self.valid_loader)
        test_loss, test_acc = self._valid(self.test_loader)

        self.logger.info('Best Valid Acc = %.4f and Final Test Acc = %.4f' % (valid_acc, test_acc))
        return test_acc

    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net

    def _load_best_net(self, net_dir):
        if net_dir is None:
            net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        self.net.load_state_dict(torch.load(net_file))

