import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=16):
        super(ConvNet, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25) # dropout
        
        # Second block
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)  # dropout
        
        # Third block
        self.conv5 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.batchnorm6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)  # dropout
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        
        # FC layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        batch_size = x.size(0)
        
        # First block
        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.relu(self.batchnorm2(self.conv2(out)))
        out = self.pool1(out)
        out = self.dropout1(out)
        
        # Second block
        out = self.relu(self.batchnorm3(self.conv3(out)))
        out = self.relu(self.batchnorm4(self.conv4(out)))
        out = self.pool2(out)
        out = self.dropout2(out)
        
        # Third block
        out = self.relu(self.batchnorm5(self.conv5(out)))
        out = self.relu(self.batchnorm6(self.conv6(out)))
        out = self.pool3(out)
        out = self.dropout3(out)
        
        out = self.adaptive_pool(out)
        
        # flatten
        out = out.view(out.size(0), -1)
        
        # FC layer
        out = self.fc1(out)
        
        # Batch normalization
        if batch_size > 1:
            out = self.fc_bn(out)
        else:
            self.fc_bn.eval()
            with torch.no_grad():
                out = self.fc_bn(out)
            self.fc_bn.train()
        
        out = self.relu(out)
        out = self.fc_dropout(out)
        out = self.fc2(out)
        
        return out