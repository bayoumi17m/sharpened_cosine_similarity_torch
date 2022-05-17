import os
import random
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from cifar import CIFAR10_1
import torchvision.transforms as transforms
from tqdm import tqdm

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

from densenet import DenseNet
from demo_network import DemoNetwork
from vgg import *
import argparse

########## Hyper Parameters ##########

batch_size = 64
n_epochs = 100

########## Setup ##########

parser = argparse.ArgumentParser(description='SCS Train')
parser.add_argument('--model', default='', help='model type')
parser.add_argument('--modelpath', default='', help='model type')
args = parser.parse_args()

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

set_all_seeds(621)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########## Model Definitions ##########

def gen_densenet_model():
    return DenseNet(sharpened_cosine_similarity=True)

def gen_demo_network():
    return DemoNetwork()

network_gen = {
    "densenet": gen_densenet_model,
    "demo": gen_demo_network,
    "vgg": vgg,
    "vgg_scs_bn_act_do": vgg_scs_bn_act_do,
    "vgg_scs_bn_do": vgg_scs_bn_do,
    "vgg_scs_bn_act": vgg_scs_bn_act,
    "vgg_scs_bn_act_do_abspool": vgg_scs_bn_act_do_abspool,
    "vgg_scs_bn_abspool": vgg_scs_bn_abspool,
    "vgg_scs_bn_act_abspool": vgg_scs_bn_act_abspool,
    "vgg_scs_bn": vgg_scs_bn,
}

model_gen = network_gen.get(args.model)
network = model_gen()
network.load_state_dict(torch.load(args.modelpath))
network.eval()

########## Data ##########

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

testing_set = CIFAR10_1(
    root=os.path.join('.', 'data', 'CIFAR10_1'),
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))

testing_loader = DataLoader(
    testing_set,
    batch_size=batch_size,
    shuffle=False)

########## Testing ##########
network = network.to(device)
epoch_testing_loss = 0
epoch_testing_num_correct = 0
with torch.no_grad():
    for batch in testing_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        epoch_testing_loss += loss.item() * testing_loader.batch_size
        epoch_testing_num_correct += (preds.argmax(dim=1).eq(labels).sum().item())

    testing_loss = epoch_testing_loss / len(testing_loader.dataset)
    testing_accuracy = (epoch_testing_num_correct / len(testing_loader.dataset))

print('Test Loss {} Test Accuracy {}'.format(testing_loss, testing_accuracy))
