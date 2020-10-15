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
from dataloader import *
import os
os.chdir('..')
from torchvision import transforms
import shutil
from models.vgg import VGG11


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
    torch.save(state, filename)

if __name__ == '__main__':
    '''load cifar10 dataset'''
    trainloader, testloader, classes = load_cifar_data()

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
