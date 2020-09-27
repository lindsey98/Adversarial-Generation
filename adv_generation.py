
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
# import torchvision.models as models
import os
from torchvision import transforms
import shutil
from utils import VGG11

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total = 0
    correct = 0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)
        target_var = target

        # compute output
        output, _ = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        _, predicted = output.max(1)
        losses.update(loss.item(), input.size(0))
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc:.3f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=correct / total))


def test(test_loader, model, epoch):
    """
        Run one train epoch
    """
    total = 0
    correct = 0

    # switch to train mode
    model.eval()

    for i, (input, target) in enumerate(test_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output, _ = model(input)

        # measure accuracy and record loss
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print('Epoch: [{0}]\t'
          'Accuracy {acc:.3f}'.format(
        epoch, acc=correct / total))

    return correct / total

def save_checkpoint(state,filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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

    '''define model'''
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the network
    model = VGG11()
    model.to(device)

    '''training'''
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=30)
    criterion = nn.CrossEntropyLoss()

    global best_prec1
    best_prec1 = 0
    for epoch in range(50):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(trainloader, model, criterion, optimizer, epoch)
        perc = test(testloader, model, epoch)
        lr_scheduler.step()
        if perc > best_prec1:
            save_checkpoint({
                'state_dict': model.state_dict()
            }, filename=os.path.join('./checkpoints/', 'model.th'))
            best_prec1 = perc

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

    shutil.rmtree('./data/normal/')
    shutil.rmtree('./data/adversarial/')
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
