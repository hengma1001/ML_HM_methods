import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
# from torchvision.utils import save_image

from vae_conv import conv_variational_autoencoder 
from data import DatasetCM, triu_to_full
import matplotlib.pyplot as plt 
import sys, os, tables 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

h5_file = '../../2F4K-0/data/2f4k_data.h5' 

cm_dataset = DatasetCM(h5_file, transform=transforms.ToTensor()) 
train_set, test_set = torch.utils.data.random_split(cm_dataset, (int(0.8*len(cm_dataset)), int(0.2*len(cm_dataset)))) 


batch_size = 128 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True) 
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)   

channels = cm_dataset.shape[-1]
# batch_size = cm_train.shape[0]/100
conv_layers = 4
feature_maps = [64,64,64,64]
filter_shapes = [3,3,3,3]
strides = [1,2,1,1]
dense_layers = 2
dense_neurons = [128, 64]
dense_dropouts = [0.0, 0.0]
latent_dim = 3

image_size = cm_dataset.shape[1] 

autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
                                           filter_shapes,strides,dense_layers,
                                           dense_neurons,dense_dropouts,latent_dim, log_interval = 10000) 

from torchsummary import summary 
summary(autoencoder.model, (1, 36, 36))


epochs = 10

for epoch in range(1, epochs + 1): 
    autoencoder.train(train_loader, epoch) 
    autoencoder.test(test_loader, epoch) 