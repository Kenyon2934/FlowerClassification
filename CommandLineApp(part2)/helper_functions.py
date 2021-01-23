import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import path
import json

# Training Functions
def get_training_args():
    parser = argparse.ArgumentParser(description="Get train.py arguments")
    parser.add_argument('data_directory', type=str, help='directory for where the data is located')
    parser.add_argument('--save_dir', type=str, default='/home/workspace/ImageClassifier/save_dir')
    parser.add_argument('--arch', default='vgg13')
    parser.add_argument('--learning_rate', default=.003, type=float)
    parser.add_argument('--hidden_units', default=[7000, 500])
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--gpu', action='store_true')
    return parser

def check_args(data_directory, save_dir, arch, learning_rate, hidden_units, epochs):
    def is_number(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    if not path.exists(data_directory):
        print(f"Error: data_directory does not exist: {data_directory}")
        exit()
    if not path.exists(save_dir):
        print(f"Error: save_dir does not exist: {data_directory}")
        exit()
    if not hasattr(models, arch):
        print(f"Error: arch value {arch} is not an existing attribute of the models module.")
        print("Please see here {https://pytorch.org/docs/stable/torchvision/models.html} for acceptable arch values")
        exit()
    if not isinstance(learning_rate, float) and not is_number(learning_rate):
        print(f"Error: learning_rate specified is nonumeric ({learning_rate}). Please use a numeric learning_rate")
        exit()
    if not isinstance(hidden_units, list):
        hidden_units = hidden_units.split(',')
    for i in hidden_units:
        if not isinstance(i,int) and not i.isnumeric():
            print(f'Error: Only use int values in list. Received {hidden_units}.')
            print('..... values should be whole numbers separated by a comma (,).')
            print('..... e.g. 4000,500  (no square brackets)')
            exit()
    if not isinstance(epochs, int):
        if isinstance(epochs, str) and not epochs.isnumeric():
            print(f"Error: badvalue: ({epochs}): epochs need to be int values only")
            exit()

def get_loaders(data_dir):
    train_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(validation_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 64)
    
    return trainloader, testloader, validationloader, train_data

def define_untrained_network(model, input_size, hidden_sizes, output_size, learning_rate, device):
    multiple_hidden_layers = len(hidden_sizes)>1
    
    sizes = []
    sizes.append(input_size)
    for i in hidden_sizes:
        sizes.append(i)
    sizes.append(output_size)
    sizes_len = len(sizes) #4
    
    od = OrderedDict()
    for i in range(sizes_len-1):
        od.update({f'fc{i+1}': nn.Linear(sizes[i], sizes[i+1])})
        if i!=(sizes_len-2):
            od.update({f'relu{i+1}': nn.ReLU()})
            od.update({f'drop{i+1}': nn.Dropout(0.2)})
        else:
            od.update({'output': nn.LogSoftmax(dim=1)})
       
    classifier = nn.Sequential(od)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    return model, criterion, optimizer

def train_model(model, epochs, trainloader, validationloader, device, optimizer, criterion):
    print('TRAINING MODEL')
    steps = 0 
    running_loss = 0
    print_every = 20
    
    for e in range(epochs):
        for inputs, labels, in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss = batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")    
                running_loss = 0
                model.train()
    print('done training model!')
    return model

def test_trained_model(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

    model.train()
    return model

def save_model_architecture_needs(train_data, num_of_epochs, optimizer, model, save_dir, arch, file_name):
    print('saving architecture needs.')
    model_architecture_needs = {
        'class_to_idx': train_data.class_to_idx,
        'num_of_epochs': num_of_epochs,
        'optmizer_state': optimizer.state_dict,
        'model_classifier': model.classifier,
        'arch': arch
    }
    
    torch.save(model_architecture_needs, f'{save_dir}/{file_name}')
    print(f'Model architecture needs successfully saved to location: {save_dir}/{file_name}')
    
def save_model_checkpoint(model, save_dir, file_name):
    print('saving model...')
    torch.save(model.state_dict(), f'{save_dir}/{file_name}')
    print(f'Model successfully saved to location: {save_dir}/{file_name}')
    
# Prediction Functions
def get_prediction_args():
    parser = argparse.ArgumentParser(description="Get predict.py arguments")
    parser.add_argument('img_path', type=str, help='directory for where the image is located')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint directory from train.py e.g. `/path/to/checkpoint`')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    return parser

def load_checkpoint(pth):
    return torch.load(pth)

def load_saved_model(model_pth, model_arch_needs_pth):
    model_arch_needs = load_checkpoint(model_arch_needs_pth)
    arch = model_arch_needs.get('arch')
    model = getattr(models, arch)(pretrained=True)
    model.classifier = model_arch_needs.get('model_classifier')
    model.load_state_dict(load_checkpoint(model_pth))
    for param in model.parameters():
        param.requires_grad = False
    
    print('Saved model successfully loaded.')
    return model, model_arch_needs
    
def process_image(im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyspTorch model
    
    #Resize Image so that shorter side is 256px, maintaining aspect ratio
    hwr = im.height/im.width 
    if im.height<=im.width:
        new_height = 256
        new_width = int(round(new_height/hwr,0))
    else:
        new_width = 256
        new_height = int(round(new_width*hwr,0))
    im_resized = im.resize((new_width, new_height))
    
    #center crop image
    targ_dim = 224
    hc = new_width - targ_dim #horizontal crop
    vc = new_height - targ_dim #vertical crop
    x1 = 0 + hc/2
    x2 = new_width - hc/2 
    y1 = 0 + vc/2
    y2 = new_height - vc/2

    im_cropped = im_resized.crop((x1,y1,x2,y2))
    
    # Normalize Color Channels
    means = np.array([0.485, 0.456, 0.406])
    st_dv = np.array([0.229, 0.224, 0.225])

    np_image = np.array(im_cropped)
    #change values from 0-255 to values between 0-1
    np_image = np_image/255

    np_image = (np_image - means) /st_dv

    # make color channel the first dimension 
    np_image = np.transpose(np_image,(2,0,1))
    
    return np_image

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        model.eval()
        model.to(device)
        img = Image.open(image_path)
        processed_img = process_image(img)
        tensor_img = torch.from_numpy(processed_img)
        tensor_img = tensor_img.float()
    
        ps = torch.exp(model(tensor_img.unsqueeze_(0).to(device)))
    top_p, top_class = ps.topk(topk, dim=1)

    model.train()
    return top_class, top_p

def process_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def derive_flower_name(cls_to_idx, top_cls, json_map):
    idx_to_cls = dict([(value, key) for key, value in cls_to_idx.items()])
    top_cls = idx_to_cls[top_cls]
    flower_name = json_map[str(top_cls)]
    return flower_name

def display_top_k_flower_and_probs(image_path, model, device, topk, model_arch_needs, category_names):
    top_cls, top_p = predict(image_path, model, device, topk)

    top_classes = []
    top_classes.extend(top_cls[0][:(topk)].cpu().numpy())
    top_probs = []
    top_probs.extend(top_p[0][:(topk)].cpu().numpy())

    flowers = []
    class_to_idx = model_arch_needs['class_to_idx']
    cat_to_name = process_json(category_names)
    
          
    for i in range(len(top_classes)):
        flowers.extend([derive_flower_name(class_to_idx, top_classes[i], cat_to_name)])
    print('------------------------------------------------')
    print(f'Top {topk} flowers and their probabilities:')
    for i in range(len(flowers)):
        print(f'Flower: {flowers[i]} | p: {top_probs[i]}')
    print('------------------------------------------------')
        