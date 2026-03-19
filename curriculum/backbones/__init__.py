from .convnet import *
from .resnet import *


def get_net(net_name, data_name):
    net_dict = {
        'convnet': ConvNet,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50
    }

    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))

    classes_dict = {
        'rvl': 16
    }
    
    return net_dict[net_name](num_classes=classes_dict[data_name])