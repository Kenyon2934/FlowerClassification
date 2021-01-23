# train.py
# Basic usage: python train.py data_directory
# Prints aout the training loss,  validation lss, and validation accuracy as the tework trains
# Parameter Options
#   data_directory
#   --save_dir (save_directory)
#   --arch (choose architecture ("vgg13"))
#   Hyperparameters...
#       --learning_rate
#       --hidden_units
#       --epochs
#   --gpu (for training)train

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

import argparse
import helper_functions as hf



# Get arguments #####
parser = hf.get_training_args()
args = parser.parse_args()

args = vars(args)
data_directory = args['data_directory'] 
save_dir = args['save_dir'] 
arch = args['arch'] 
learning_rate =args['learning_rate'] 
hidden_units =args['hidden_units'] 
epochs =args['epochs'] 
gpu=args['gpu'] 

hf.check_args(data_directory, save_dir, arch, learning_rate, hidden_units, epochs)
hidden_units =( hidden_units if isinstance(hidden_units, list) else hidden_units.split(',') )
hidden_units = [int(i) for i in hidden_units]

device = torch.device("cuda" if gpu else "cpu")



#get data loaders
trainloader, testloader, validationloader, train_data = hf.get_loaders(data_directory)

#get cat_to_name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
          
          
# Building and training the classifier #################
model = getattr(models, arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

#Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
model, criterion, optimizer = hf.define_untrained_network(model, model.classifier[0].in_features, hidden_units, len(cat_to_name), learning_rate, device) #input_size, hidden_sizes, output_size

# Train Model
trained_model = hf.train_model(model, epochs, trainloader, validationloader, device, optimizer, criterion)

# Test Trained Model
hf.test_trained_model(trained_model, testloader, criterion, device)

# Save
hf.save_model_architecture_needs(train_data, epochs, optimizer, model, save_dir, arch, 'model_architecture_needs.pth')
hf.save_model_checkpoint(model, save_dir, 'model_checkpoint.pth')