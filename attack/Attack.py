import numpy as np
import copy
import os
os.chdir('..')
from dataloader import *
from models.vgg import VGG11
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable



class adversarial_attack():
    
    def __init__(self, method, model, dataloader, device, num_classes=10, save_data=False):
        self.method = method
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.save_data = save_data
        
    def batch_attack(self):
        '''
        Run attack on a batch of data
        
        '''
        # Accuracy counter
        correct = 0
        total = 0
        adv_examples = []
        ct_save = 0
        adv_cat = torch.tensor([])

        # Loop over all examples in test set
        for data, label in self.dataloader:
            data, label = data.to(self.device), label.to(self.device)

            # Forward pass the data through the model
            output, _ = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            if init_pred.item() != label.item():  # initially was incorrect --> no need to generate adversary
                continue

            # Call Attack
            if self.method in ['fgsm', 'stepll']:
                criterion = nn.CrossEntropyLoss()
                perturbed_data = self._FGSM(data, label, criterion, epsilon=0.05, clip_min=0.0, clip_max=1.0)
                
            elif self.method == 'jsma':
                # randomly select a target class
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, output.size()[1], (1,)).to(self.device)
                perturbed_data = self._JSMA(data, target_class, max_iter=100, clip_min=0.0, clip_max=1.0)
                
            elif self.method == 'deepfool':
                f_image = output.detach().cpu().numpy().flatten()
                I = (np.array(f_image)).flatten().argsort()[::-1]
                perturbed_data = self._deep_fool(data, label, I, overshoot=0.02, max_iter=100)
                
            elif self.method == 'cw':
                perturbed_data = self._cw(data, label, targeted=False, c=1e-4, kappa=0, max_iter=500, learning_rate=0.01)
                
            else:
                print('Attack method is not supported')
                
            # Re-classify the perturbed image
            self.model.eval()
            with torch.no_grad():
                output, _ = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == label.item():
                correct += 1  # still correct
            else:# save successful attack
                if ct_save <= 100:
                    if self.save_data:
                        os.makedirs('./data/normal_{}'.format(attack_method), exist_ok=True)
                        os.makedirs('./data/adversarial_{}'.format(attack_method), exist_ok=True)
                        # Save the original instance
                        torch.save((data.detach().cpu(), init_pred.detach().cpu()),
                                   './data/normal_{}/{}.pt'.format(attack_method, ct_save))
                        # Save the adversarial example
                        torch.save((perturbed_data.detach().cpu(), final_pred.detach().cpu()),
                                   './data/adversarial_{}/{}.pt'.format(attack_method, ct_save))
                ct_save += 1


#             else:  # attack is successful, final class is cat
#                 if final_pred.item() == 3:
#                     adv_cat = torch.cat([adv_cat, perturbed_data.detach().cpu()], dim=0)
#                     torch.save(adv_cat, './data/adv_cat_%s.pt'%self.method)

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            total += 1

        # Calculate final accuracy
        final_acc = correct / float(len(self.dataloader))
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
        
        
        
    def _FGSM(self, image, label, criterion, epsilon=0.05, clip_min=0.0, clip_max=1.0):
        ''' https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
        FGSM attack
        Parameters:
            image: input image
            label: gt label
            criterion: loss function to use
            epsilon: perturbation strength
            clip_min, clip_max: minimum/maximum value a pixel can take
        '''
        
        pert_image = copy.deepcopy(image)
        x = Variable(pert_image, requires_grad=True)

        output, _ = self.model(x)

        if self.method == 'fgsm':
            loss = criterion(output, label)  # loss for ground-truth class
        else:
            ll = output.min(1, keepdim=True)[1][0]
            loss = criterion(output, ll)  # Loss for least-likely class

        # Back propogation
        self.model.zero_grad()
        loss.backward()

        # Collect the sign of the data gradient
        sign_data_grad = torch.sign(x.grad.data)

        # Create the perturbed image by adjusting each pixel of the input image
        if self.method == 'fgsm':
            perturbed_image = image + epsilon * sign_data_grad
        else:
            perturbed_image = image - epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, clip_min, clip_max)

        return perturbed_image
        
    def _JSMA(self, image, target, max_iter=100, clip_min=0.0, clip_max=1.0):
        '''https://github.com/ast0414/adversarial-example/blob/master/craft.py
        Saliency map attack
        Parameters:
            image: input image
            target: target class
            max_iter: maximum iteration
            clip_min: minimum value of pixel
            clip_max: maximum value of pixel
        Returns:
            perturbed image
        '''
        
        def compute_jacobian(inputs, output):
            '''Helper function: compute jacobian matrix of confidence score vector w.r.t. input'''

            jacobian = torch.zeros(self.num_classes, *inputs.size()).cuda()

            for i in range(self.num_classes):
                zero_gradients(inputs)
                output[0, i].backward(retain_graph=True)
                jacobian[i] = inputs.grad.data

            return torch.transpose(jacobian, dim0=0, dim1=1)


        def saliency_map(jacobian, search_space, target_index):
            '''Helper function: compute saliency map and select the maximum index'''
            jacobian = jacobian.squeeze(0)
            alpha = jacobian[target_index]
            beta = jacobian.sum(0) - alpha

            mask1 = torch.ge(alpha, 0.0)
            mask2 = torch.le(beta, 0.0)

            mask = torch.mul(torch.mul(mask1, mask2), search_space)

            saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
            saliency_map = saliency_map.sum(0).sum(0)

            row_idx, col_idx = (saliency_map == torch.max(saliency_map)).nonzero()[0]
            return row_idx, col_idx


        # Make a clone since we will alter the values
        pert_image = copy.deepcopy(image)
        x = Variable(pert_image, requires_grad=True)

        output, _ = self.model(x)
        _, label = torch.max(output.data, 1)
        
        count = 0
        # if attack is successful or reach the maximum number of iterations
        while (count < max_iter) and (label != target):

            search_space = (x.data[0] > clip_min) & (x.data[0] < clip_max)

            # Calculate Jacobian
            jacobian = compute_jacobian(x, output)

            # get the highest saliency map's index
            row_idx, col_idx = saliency_map(jacobian, search_space, target)

            # increase to its maximum value
            x.data[0, :, row_idx, col_idx] = clip_max

            # recompute prediction
            output, _ = self.model(x)
            label = output.max(1, keepdim=True)[1]

            count += 1

        return x.data
    
    
    
    def _deep_fool(self, image, label, I, overshoot=0.02, max_iter=100):
        '''https://github.com/LTS4/DeepFool/tree/master/Python
        DeepFool attack
        Parameters:
            image: input image
            label: ground-truth label
            I: current predicted class ranked by decending order
            overshoot: scale factor to increase perturbation a little bit
            max_iter: maximum iterations allowed
        Returns:
            perturbed image
        '''

        pert_image = copy.deepcopy(image)
        w = np.zeros(image.shape)
        r_tot = np.zeros(image.shape)

        loop_i = 0

        x = Variable(pert_image, requires_grad=True)
        fs, _ = self.model(x)
        fs_list = [fs[0, I[k]] for k in range(self.num_classes)]
        k_i = label

        # Stop until attack is successful or reach the maximum iterations
        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True) # backpropogate the maximum confidence score
            grad_orig = x.grad.data.detach().cpu().numpy().copy()

            for k in range(1, self.num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.detach().cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.detach().cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()

            x = Variable(pert_image, requires_grad=True)
            fs, _ = self.model(x)
            k_i = np.argmax(fs.data.detach().cpu().numpy().flatten())

            loop_i += 1

        return x.data
    
    def _cw(self, image, label, targeted=False, c=1e-4, kappa=0, max_iter=500, learning_rate=0.01):
        '''https://github.com/Harry24k/CW-pytorch/blob/master/CW.ipynb
           Launch C&W-L2 attack for a single image
           Parameters:
               image: input image
               label: gt label
               targeted: targeted attack or untargeted attack, default is False
               c: trade-off parameter 
               kappa: margin
               max_iter: maximum iterations allowed for attack
               learning_rate: learning rate in optimization
           Returns:
               perturbed image
        '''
        image, label = image.to(self.device), label.to(self.device)

        # Define helper function for c&w attack
        def f(x):
            '''Return the c'''
            output, _ = self.model(x)
            one_hot_label = torch.eye(len(output[0]))[label].to(self.device)

            i, _ = torch.max((1-one_hot_label)*output, dim=1) # maximum confidence for other class
            j = torch.masked_select(output, one_hot_label.byte().bool()) # confidence for ground-truth class

            # If targeted, optimize for making the other class most likely 
            if targeted:
                return torch.clamp(i-j, min=-kappa)

            # If untargeted, optimize for making the other class most likely 
            else:
                return torch.clamp(j-i, min=-kappa)

        w = torch.zeros_like(image, requires_grad=True).to(self.device)
        optimizer = optim.Adam([w], lr=learning_rate)
        prev = 1e10

        # Stop until reach the maximum iteration or loss starts diverging
        for step in range(max_iter) :

            a = 1/2*(nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction='sum')(a, image)
            loss2 = torch.sum(c*f(a))

            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (max_iter//10) == 0:
                if cost > prev:
                    print('Attack Stopped since loss starts increasing....')
                    return a
                prev = cost

            print('- Learning Progress : %2.2f %%' %((step+1)/max_iter*100), end='\r')

        attack_image = 1/2*(nn.Tanh()(w) + 1)

        return attack_image
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="Attack method you want to use", required=True)
    parser.add_argument("--save", help="Whether to save adversarial and normal data", default='False')
    args = parser.parse_args()
    
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
    model = model.eval()
    print("Successfully load pre-trained model")
    
    print("Start attack with method: {}".format(args.method))
    if args.save == 'True':
        check = adversarial_attack(method=args.method, model=model, dataloader=testloader, device=device, num_classes=10, save_data=True)
    else:
        check = adversarial_attack(method=args.method, model=model, dataloader=testloader, device=device, num_classes=10)
        
    check.batch_attack()
    
    print("Finish attack")