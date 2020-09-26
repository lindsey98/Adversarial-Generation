import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
import os

# Initialize the network

def load_model(weights_path):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 10)
    
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    model.eval()
    return model

def partial_model_execute(model, layer_ct):
    partial_conv = nn.Sequential(*list(model.children())[:layer_ct])
    for param in partial_conv.parameters():
        param.requires_grad = False
        
    return partial_conv

def load_cifar_data():
    # CIFAR10 Test dataset and dataloader declaration
    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


def load_samples(path):
    inputs_list = torch.tensor([])
    targets_list = torch.tensor([], dtype=torch.long)
    
    for file in os.listdir(path):
        if file.endswith(".pt"):
            inputs, targets = torch.load(os.path.join(path, file))
            inputs = inputs.detach().cpu()
            targets = targets.detach().cpu()
            
            inputs_list = torch.cat([inputs_list, inputs], dim=0)
            targets_list = torch.cat([targets_list, targets], dim=0)
            
    return inputs_list, targets_list
            