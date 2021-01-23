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




# Get arguments
parser = hf.get_prediction_args()

args = parser.parse_args()
args = vars(args)
img_path = args['img_path'] 
checkpoint = args['checkpoint']
top_k = args['top_k'] 
category_names = args['category_names'] 
gpu=args['gpu'] 

device = torch.device("cuda" if gpu else "cpu")

model_pth = f'{checkpoint}/model_checkpoint.pth'
model_arch_needs_pth =  f'{checkpoint}/model_architecture_needs.pth'
model, model_arch_needs = hf.load_saved_model(model_pth, model_arch_needs_pth)

hf.display_top_k_flower_and_probs(img_path, model, device, top_k, model_arch_needs, category_names)
