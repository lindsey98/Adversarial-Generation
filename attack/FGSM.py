
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
import time
import os
from torchvision import transforms
import shutil
from dataloader import *
from models.vgg import VGG11

def adv_attack(image, epsilon, data_grad, attack_method):
    assert attack_method in ['fgsm', 'stepll']

    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)

    # Create the perturbed image by adjusting each pixel of the input image
    if attack_method == 'fgsm':
        perturbed_image = image + epsilon * sign_data_grad
    else:
        perturbed_image = image - epsilon * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def fgsm_attack(model, device, dataloader, criterion, attack_method, epsilon):
    assert attack_method in ['fgsm', 'stepll']

    # Accuracy counter
    correct = 0
    total = 0
    adv_examples = []
    ct_save = 0

    # Loop over all examples in test set
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output, _ = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if init_pred.item() != target.item():  # initially was incorrect --> no need to generate adversary
            continue

        if attack_method == 'fgsm':
            loss = criterion(output, target)  # loss for ground-truth class
        else:
            ll = output.min(1, keepdim=True)[1][0]
            loss = criterion(output, ll)  # Loss for least-likely class

        # Back propogation
        model.zero_grad()
        loss.backward()

        # Collect data_grad
        data_grad = data.grad.data

        # Call Attack
        perturbed_data = adv_attack(data, epsilon, data_grad, attack_method)

        # Re-classify the perturbed image
        model.eval()
        with torch.no_grad():
            output, _ = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1  # still correct

        else:  # attack is successful
            if ct_save < 100:
                os.makedirs('./data/normal', exist_ok=True)
                os.makedirs('./data/adversarial', exist_ok=True)
                # Save the original instance
                torch.save((data.detach().cpu(), init_pred.detach().cpu()),
                           './data/normal/{}_{}.pt'.format(attack_method, ct_save))
                # Save the adversarial example
                torch.save((perturbed_data.detach().cpu(), final_pred.detach().cpu()),
                           './data/adversarial/{}_{}.pt'.format(attack_method, ct_save))
            ct_save += 1

        # Special case for saving 0 epsilon examples
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        total += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(dataloader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == '__main__':
    '''load cifar10 dataset'''
    trainloader, testloader, classes = load_cifar_data()

    '''define model'''
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the network
    model = VGG11()
    model.to(device)

    '''load saved model'''
    model.load_state_dict(torch.load('./checkpoints/model.th')['state_dict'])
    model.eval()

    '''generate and save adversarial examples'''
    accuracies = []
    examples = []
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    attack_method = 'fgsm'

    # for eps in epsilons:
    #     acc, ex = test(model, device, testloader, criterion, attack_method, eps)
    #     accuracies.append(acc)
    #     examples.append(ex)
    
    try:
        shutil.rmtree('./data/normal/')
        shutil.rmtree('./data/adversarial/')
    except:
        pass
    acc, ex = fgsm_attack(model, device, testloader, criterion, attack_method, epsilon=0.05)

    '''visualize testing accuracy decrease'''
    # plt.figure(figsize=(5, 5))
    # plt.plot(epsilons[:len(accuracies)], accuracies, "*-")
    # # plt.yticks(np.arange(0, 1.1, step=0.1))
    # # plt.xticks(np.arange(0, .35, step=0.05))
    # plt.title("Accuracy vs Epsilon")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.show()
