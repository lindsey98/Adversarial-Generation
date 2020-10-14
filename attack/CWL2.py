import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import copy

def reduce_sum(x, keepdim=True):
    '''Helper function Perform sum on all dimension except for batch dim
    '''
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x



def l2_dist(x, y, keepdim=True):
    '''Helper function Compute L2-dist
    '''
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)



def torch_arctanh(x, eps=1e-6):
    '''Helper function Implement arctanh function
    '''
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5



def tanh_rescale(x, x_min=0., x_max=1.):
    '''Helper function Implement tanh function
    '''
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


def compare(output, target):
    '''Helper function Compare predicted value with ground-truth
    '''
    if not isinstance(output, (float, int, np.int64)):
        output = np.copy(output)
        output[target] -= confidence
        output = np.argmax(output)
    return output == target



def cal_loss(output, target, dist, scale_const):
    '''Helper function Compute loss for C&W L2
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




def optimize(model, optimizer, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
    '''Helper function
       Optimize C&W L2 loss by Adam
    '''
    # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
    input_adv = tanh_rescale(modifier_var + input_var, clip_min, clip_max)
    output = model(x)

    # distance to the original input data
    if input_orig is None:
        dist = l2_dist(input_adv, input_var, keepdim=False)
    else:
        dist = l2_dist(input_adv, input_orig, keepdim=False)

    loss = cal_loss(output, target_var, dist, scale_const_var)

    optimizer.zero_grad()
    model.zero_grad()
    loss.backward()
    optimizer.step()

    loss_np = loss.item()
    dist_np = dist.data.detach().cpu().numpy()
    output_np = output.data.detach().cpu().numpy()
    input_adv_np = input_adv.data # back to BHWC for numpy consumption
    return loss_np, dist_np, output_np, input_adv_np




def cw(model, num_classes, device, image, target, max_steps=1000, clip_min=-1.0, clip_max=1.0):
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
    target_onehot = torch.zeros(target.size() + (num_classes,)).to(device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    # setup the modifier variable, this is the variable we are optimizing over
    modifier = torch.zeros(input_var.size()).float()
    modifier = modifier.to(device)
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
        scale_const_tensor = scale_const_tensor.to(device)
        scale_const_var = Variable(scale_const_tensor, requires_grad=False)

        prev_loss = 1e6
        for step in range(max_steps):
            # perform the attack
            loss, dist, output, adv_img = optimize(
                model,
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

