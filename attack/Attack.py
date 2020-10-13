import numpy as np
import copy
import os
os.chdir('..')
from dataloader import *
from models.vgg import VGG11
import argparse
import sys

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
        for ct, (data, label) in enumerate(self.dataloader):
            data, label = data.to(self.device), label.to(self.device)

            # Forward pass the data through the model
#             output, _ = self.model(data)
            output = self.model(data)
            self.model.zero_grad()
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            if init_pred.item() != label.item():  # initially was incorrect --> no need to generate adversary
                total += 1
                print(ct)
                continue

            # Call Attack
            if self.method in ['fgsm', 'stepll']:
                criterion = nn.CrossEntropyLoss()
                perturbed_data = self._FGSM(data, label, criterion)
                
            elif self.method == 'jsma':
                # randomly select a target class
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, output.size()[1], (1,)).to(self.device)
                perturbed_data = self._JSMA(data, target_class)
                
            elif self.method == 'deepfool':
                f_image = output.detach().cpu().numpy().flatten()
                I = (np.array(f_image)).flatten().argsort()[::-1]
                perturbed_data = self._deep_fool(data, label, I)
                
            elif self.method == 'cw':
                # randomly select a target class
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, output.size()[1], (1,)).to(self.device)
                perturbed_data = self._cw(data, target_class, max_steps=1000)
                
            else:
                print('Attack method is not supported')
                
            self.model.zero_grad()
            # Re-classify the perturbed image
            self.model.eval()
            with torch.no_grad():
#                 output, _ = self.model(perturbed_data)
                output = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == init_pred.item():
                correct += 1  # still correct
            else:# save successful attack
                print(final_pred)
                print(init_pred)
                if self.save_data:
                    os.makedirs('./data/normal_{}'.format(self.method), exist_ok=True)
                    os.makedirs('./data/adversarial_{}'.format(self.method), exist_ok=True)
                    # Save the original instance
                    torch.save((data.detach().cpu(), init_pred.detach().cpu()),
                               './data/normal_{}/{}.pt'.format(self.method, ct_save))
                    # Save the adversarial example
                    torch.save((perturbed_data.detach().cpu(), final_pred.detach().cpu()),
                               './data/adversarial_{}/{}.pt'.format(self.method, ct_save))

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            total += 1
            print(ct)
            

        # Calculate final accuracy
        final_acc = correct / float(len(self.dataloader))
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
        
        
        
    def _FGSM(self, image, label, criterion, max_iter=100, epsilon=0.05, clip_min=-1.0, clip_max=1.0):
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

        output = self.model(x)
        pred = output.max(1, keepdim=True)[1]
        iter_ct = 0
        
        while pred == label:
            if self.method == 'fgsm':
                loss = criterion(output, label)  # loss for ground-truth class
            else:
                ll = output.min(1, keepdim=True)[1][0]
                loss = criterion(output, ll)  # Loss for least-likely class

            # Back propogation
            zero_gradients(x)
            self.model.zero_grad()
            loss.backward()

            # Collect the sign of the data gradient
            sign_data_grad = torch.sign(x.grad.data.detach())

            # Create the perturbed image by adjusting each pixel of the input image
            if self.method == 'fgsm':
                x.data = x.data + epsilon * sign_data_grad
            else:
                x.data = x.data - epsilon * sign_data_grad

            # Adding clipping to maintain [0,1] range
            
            x.data = torch.clamp(x.data, clip_min, clip_max)
            output = self.model(x)
            pred = output.max(1, keepdim=True)[1]
            
            iter_ct += 1
            if iter_ct >= max_iter:
                break

        return x.data
        
    def _JSMA(self, image, target, max_iter=100, clip_min=-1.0, clip_max=1.0):
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
                self.model.zero_grad()
                output[0, i].backward(retain_graph=True)
                jacobian[i] = inputs.grad.data

            return torch.transpose(jacobian, dim0=0, dim1=1)


        def saliency_map(jacobian, search_space, target_index):
            '''Helper function: compute saliency map and select the maximum index'''
            jacobian = jacobian.squeeze(0)
            alpha = jacobian[target_index].sum(0).sum(0)
            beta = jacobian.sum(0).sum(0) - alpha
            
            # filter by the sign of alpha and beta
            mask1 = torch.ge(alpha, 0.0)
            mask2 = torch.le(beta, 0.0)
            mask = torch.mul(torch.mul(mask1, mask2), search_space)
            saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
            
            # get the maximum index
            row_idx, col_idx = (saliency_map == torch.max(saliency_map)).nonzero()[0]
            return row_idx, col_idx


        # Make a clone since we will alter the values
        pert_image = copy.deepcopy(image)
        x = Variable(pert_image, requires_grad=True)

        output = self.model(x)
        label = output.max(1, keepdim=True)[1]
        
        count = 0
        # if attack is successful or reach the maximum number of iterations
        while (count < max_iter) and (label != target):
            
            # Skip the pixels that have been attacked before
            search_space = (x.data[0].sum(0) > clip_min*x.data.shape[1]) & (x.data[0].sum(0) < clip_max*x.data.shape[1])

            # Calculate Jacobian
            jacobian = compute_jacobian(x, output)

            # get the highest saliency map's index
            row_idx, col_idx = saliency_map(jacobian, search_space, target)

            # increase to its maximum value
            x.data[0, :, row_idx, col_idx] = clip_max

            # recompute prediction
            output = self.model(x)
            label = output.max(1, keepdim=True)[1]

            count += 1
            if count >= max_iter:
                break

        return x.data
    
    
    
    def _deep_fool(self, image, label, I, overshoot=0.02, max_iter=100, clip_min=-1.0, clip_max=1.0):
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
        fs = self.model(x)
        fs_list = [fs[0, I[k]] for k in range(self.num_classes)]
        k_i = label

        # Stop until attack is successful or reach the maximum iterations
        while k_i == label and loop_i < max_iter:

            pert = np.inf
            zero_gradients(x)
            self.model.zero_grad()
            fs[0, I[0]].backward(retain_graph=True) # backpropogate the maximum confidence score
            grad_orig = x.grad.data.detach().cpu().numpy().copy()

            for k in range(1, self.num_classes):
                zero_gradients(x)
                self.model.zero_grad()
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.detach().cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.detach().cpu().numpy()
                
                if np.linalg.norm(w_k.flatten()) == 0.0: # if w_k is all zero, no perturbation at all
                    pert_k = 0.0 * abs(f_k)
                else:
                    pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            if np.linalg.norm(w) == 0.0:
                r_i = (pert+1e-4) * w
            else:
                r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
            pert_image = torch.clamp(pert_image, clip_min, clip_max)

            x = Variable(pert_image, requires_grad=True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.detach().cpu().numpy().flatten())

            loop_i += 1
            if loop_i >= max_iter:
                break

        return x.data
    
    def _cw(self, image, target, max_steps=1000, clip_min=-1.0, clip_max=1.0):
        '''https://github.com/rwightman/pytorch-nips2017-attack-example/blob/master/attacks/attack_carlini_wagner_l2.py
           C&W L2 attack
           Parameters:
               image: input image
               target: adv target class
               max_steps: maximum iterations of optimization
               clip_min, clip_max: clip image into legal range
        '''
        
        confidence = 20  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        initial_const = 0.1  # bumped up from default of .01 in reference code
        binary_search_steps = 5
        repeat = binary_search_steps >= 10
        abort_early = True
        debug = False
        batch_idx = 0

        def reduce_sum(x, keepdim=True):
            '''Helper function
               Perform sum on all dimension except for batch dim
            '''
            for a in reversed(range(1, x.dim())):
                x = x.sum(a, keepdim=keepdim)
            return x

        def l2_dist(x, y, keepdim=True):
            '''Helper function
               Compute L2-dist
            '''
            d = (x - y)**2
            return reduce_sum(d, keepdim=keepdim)

        def torch_arctanh(x, eps=1e-6):
            '''Helper function
               Implement arctanh function
            '''
            x *= (1. - eps)
            return (torch.log((1 + x) / (1 - x))) * 0.5

        def tanh_rescale(x, x_min=0., x_max=1.):
            '''Helper function
               Implement tanh function
            '''
            return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

        def compare(output, target):
            '''Helper function
               Compare predicted value with ground-truth
            '''
            if not isinstance(output, (float, int, np.int64)):
                output = np.copy(output)
                output[target] -= confidence
                output = np.argmax(output)
            return output == target

        def cal_loss(output, target, dist, scale_const):
            '''Helper function
               Compute loss for C&W L2
            '''
            # compute the probability of the label class versus the maximum other
            real = (target * output).sum(1)
            other = ((1. - target) * output - target * 10000.).max(1)[0]
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
            loss1 = torch.sum(scale_const * loss1)

            loss2 = dist.sum()

            loss = loss1 + loss2
            return loss


        def optimize(optimizer, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
            '''Helper function
               Optimize C&W L2 loss by Adam
            '''
            # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
            input_adv = tanh_rescale(modifier_var + input_var, clip_min, clip_max)

#             output, _ = self.model(input_adv)
            output = self.model(x)

            # distance to the original input data
            if input_orig is None:
                dist = l2_dist(input_adv, input_var, keepdim=False)
            else:
                dist = l2_dist(input_adv, input_orig, keepdim=False)

            loss = cal_loss(output, target_var, dist, scale_const_var)

            optimizer.zero_grad()
            self.model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_np = loss.item()
            dist_np = dist.data.detach().cpu().numpy()
            output_np = output.data.detach().cpu().numpy()
            input_adv_np = input_adv.data # back to BHWC for numpy consumption
            return loss_np, dist_np, output_np, input_adv_np

        batch_size = image.size(0)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = image
        
        # setup input (image) variable, clamp/scale as necessary
        # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
        # this as per the reference implementation or can we skip the arctanh?
        input_var = Variable(torch_arctanh(image), requires_grad=False)
        input_orig = tanh_rescale(input_var, clip_min, clip_max)

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,)).to(self.device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        modifier = modifier.to(self.device)
        modifier_var = Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=0.0005)

        for search_step in range(binary_search_steps):
            print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if repeat and search_step == binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            scale_const_tensor = scale_const_tensor.to(self.device)
            scale_const_var = Variable(scale_const_tensor, requires_grad=False)

            prev_loss = 1e6
            for step in range(max_steps):
                # perform the attack
                loss, dist, output, adv_img = optimize(
                    optimizer,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_orig)

                if step % 100 == 0 or step == max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                        step, loss, dist.mean(), modifier_var.data.mean()))

                if abort_early and step % (max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and compare(output_logits, target_label):
                        if debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and compare(output_logits, target_label):
                        if debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack
    
    
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