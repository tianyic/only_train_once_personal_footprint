import numpy as np
import torch

def is_conv_weights(shape):
    return len(shape) == 4

def is_linear_weights(shape, num_classes):
    if len(shape) == 2 and shape[0] != num_classes:
        return True
    else:
        return False

def compute_sparsity(weights):
    nnz = 0
    nnz_tolerance = 0
    n = 0
    num_zero_kernels = 0
    num_all_kernels = 0

    for weight in weights:
        cur_weight = weight.data.cpu().numpy()
        if is_conv_weights(cur_weight.shape):
            nnz += np.sum(cur_weight != 0)
            nnz_tolerance += np.sum(np.abs(cur_weight) > 1e-6)
            n += cur_weight.size
            for k in range(cur_weight.shape[0]):
                if np.sum(cur_weight[k,...]==0) == cur_weight[k,...].size:
                    num_zero_kernels += 1
            num_all_kernels += cur_weight.shape[0]

    return 1.0 - float(nnz) / float(n+1e-6), 1.0 - float(nnz_tolerance) / float(n+1e-6), float(num_zero_kernels) / num_all_kernels

def compute_sparsity_linear(weights, num_classes):
    nnz = 0
    nnz_tolerance = 0
    n = 0
    num_zero_kernels = 0
    num_all_kernels = 0

    for weight in weights:
        cur_weight = weight.data.cpu().numpy()
        if is_conv_weights(cur_weight.shape):
            nnz += np.sum(cur_weight != 0)
            nnz_tolerance += np.sum(np.abs(cur_weight) > 1e-6)
            n += cur_weight.size
            for k in range(cur_weight.shape[0]):
                if np.sum(cur_weight[k,...]==0) == cur_weight[k,...].size:
                    num_zero_kernels += 1
            num_all_kernels += cur_weight.shape[0]

        elif is_linear_weights(cur_weight.shape, num_classes):
            nnz += np.sum(cur_weight != 0)
            nnz_tolerance += np.sum(np.abs(cur_weight) > 1e-6)
            n += cur_weight.size
            for k in range(cur_weight.shape[0]):
                if np.sum(cur_weight[k, ...]==0) == cur_weight[k,...].size:
                    num_zero_kernels += 1
            num_all_kernels += cur_weight.shape[-1]

    return 1.0 - float(nnz) / float(n+1e-6), 1.0 - float(nnz_tolerance) / float(n+1e-6), float(num_zero_kernels) / num_all_kernels


def compute_F(trainloader, model, weights, criterion, lmbda):
    f = 0.0
    device = next(model.parameters()).device
    for index, (X, y) in enumerate(trainloader):
        X = X.to(device)
        y = y.to(device)
        y_pred = model.forward(X)
        f1 = criterion(y_pred, y) # mean at batch
        f += float(f1)
    f /= len(trainloader)
    norm_l1_x_list = []
    for w in weights:
        norm_l1_x_list.append(torch.norm(w, 1).item())
    norm_l1_x = sum(norm_l1_x_list)
    F = f + lmbda * norm_l1_x

    return F



def compute_func_values(trainloader, model, criterion, lmbda, omega):
    f = 0.0
    device = next(model.parameters()).device
    for index, (X, y) in enumerate(trainloader):
        X = X.to(device) 
        y = y.to(device) 
        y_pred = model.forward(X)
        f1 = criterion(y_pred, y)
        f += float(f1)
    f /= len(trainloader)

    F = f + lmbda * omega
    return F, f




def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


def check_accuracy(model, testloader):
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            total += y.size(0)

            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            
            correct1 += prec1.item()
            correct5 += prec5.item()

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5


