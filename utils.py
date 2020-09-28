
import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
import os
from models.vgg import VGG11
from dataloader import *

    
def load_model(weights_path, device):
    '''
      Load model
      parameters:
          weights_path: saved checkpoint
          device: cuda or cpu
          
      return:
          model with pretrained weights
    '''
    
    # use VGG11 model
    model = VGG11()
    
    # load state dictionary
    model.load_state_dict(torch.load(weights_path)['state_dict'])
    
    # push to device, disable dropout etc
    model = model.to(device).eval()
    return model

def partial_model_execute(model, layer_ct):
    '''
      Execute partial model
      parameters:
          model
          layer_ct: how many children to include in the partial model
      return:
          partial model
    '''
    partial_conv = nn.Sequential(*list(model.children())[:layer_ct])
    
    #disable gradient updates
    for param in partial_conv.parameters():
        param.requires_grad = False
        
    return partial_conv

def batch_model_execute(model, layer_ct, data, device):
    '''
      Execute partal model on a bunch of data
      parameters:
          model
          layer_ct: how many children to include in the partial model
          data: list/tensors of input images
          device: cpu or cuda
      return:
          output feature map tensor
    '''
    
    # push inputs to device
    inputs = data.to(device)
    
    # initialize partial model
    partial_model = partial_model_execute(model, layer_ct).to(device)
    partial_model.eval()
    
    # output tensor to return
    outputs = torch.tensor([])
    
    with torch.no_grad():
        for inp in inputs: # instance by instance
            inp = inp[None, ...]
            oup = partial_model(inp).detach().cpu()
            outputs = torch.cat([outputs, oup], dim=0)
            
    return outputs

            
