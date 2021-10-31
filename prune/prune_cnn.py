import sys
import time
import os 
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from backend import Model
from data.datasets import Dataset
from utils.flops import compute_flops
from utils.utils import check_accuracy


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, type=str, choices=['resnet50', 'vgg16_bn', 'vgg16'])
    parser.add_argument('--dataset_name', required=True, type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--checkpoint', required=True, type=str) 
    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()

    backend = args.backend
    dataset_name = args.dataset_name
    checkpoint = args.checkpoint
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name == "cifar10" and backend == "resnet50":
        backend = "resnet50_cifar10"
    elif dataset_name == "imagenet" and backend == "resnet50":
        backend = "resnet50_imagenet"

    print("Construct original full group sparse model")
    model = Model(backend=backend, device=device)
    cfg = model.get_config()
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    print("Get pruned model")
    prune_model, prune_cfg = model.prune()
    prune_model = prune_model.to(device)

    print("Save pruned model to: ", os.path.join("_".join(args.checkpoint.split("_")[:-2])+'_pruned.pt'))
    torch.save(prune_model.state_dict(), os.path.join("_".join(args.checkpoint.split("_")[:-2])+'_pruned.pt'))

    orig_num_params, orig_num_flops = compute_flops(dataset_name, backend, cfg=cfg)
    prune_num_params, prune_num_flops = compute_flops(dataset_name, backend, cfg=prune_cfg)

    print('Original #params:', orig_num_params, 'Original FLOPs:', orig_num_flops)
    print('Pruned #params:', prune_num_params, 'Pruned FLOPs:', prune_num_flops)
    print('Params Reduction: %f'%(float(prune_num_params) / float(orig_num_params)))
    print('FLOPs Reduction: %f'%( float(prune_num_flops) / float(orig_num_flops)))

    trainloader, testloader = Dataset(dataset_name, batch_size=batch_size)

    print("Evaludate full group sparse model:")
    accuracy1, accuracy5 = check_accuracy(model, testloader)
    print('Group Sparse Model Accuracy1:', accuracy1, 'Accuracy5:', accuracy5)

    print("Evaludate pruned model:")
    accuracy1, accuracy5 = check_accuracy(prune_model, testloader)
    print('Pruned Model AFTER assignning weights Accuracy1:', accuracy1, 'Accuracy5:', accuracy5)
