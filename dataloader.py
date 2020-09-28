import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
import os

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


def load_samples(path, method):
    inputs_list = torch.tensor([])
    targets_list = torch.tensor([], dtype=torch.long)
    
    count_sample = len([file for file in os.listdir(path) if file.endswith(".pt")])
    
    for ct in range(count_sample):
        filename = method + '_' + str(ct) + '.pt'
        inputs, targets = torch.load(os.path.join(path, filename))

        inputs_list = torch.cat([inputs_list, inputs], dim=0)
        targets_list = torch.cat([targets_list, targets], dim=0)

        if ct % 50 == 0:
            print(ct, " examples are loaded")
            
    return inputs_list, targets_list