import sys
sys.path.append('..')
from sanity_check.backends.resnet_cifar10 import resnet18_cifar10
from only_train_once import OTO
import torch

model = resnet18_cifar10()
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# A ResNet_zig.gv.pdf will be generated to display the depandancy graph.
oto.visualize(view=False, out_dir='../cache')

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

trainset = CIFAR10(root='cifar10', train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
testset = CIFAR10(root='cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader =  torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

optimizer = oto.hesso(
    variant='sgd', 
    lr=0.1, 
    weight_decay=1e-4,
    target_group_sparsity=0.7,
    start_pruning_step=30 * len(trainloader),
    pruning_periods=10,
    pruning_steps=30 * len(trainloader)
)

from utils.utils import check_accuracy

max_epoch = 300
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
# Every 75 epochs, decay lr by 10.0
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1) 

for epoch in range(max_epoch):
    f_avg_val = 0.0
    model.train()
    lr_scheduler.step()
    for X, y in trainloader:
        X = X.cuda()
        y = y.cuda()
        y_pred = model.forward(X)
        f = criterion(y_pred, y)
        optimizer.zero_grad()
        f.backward()
        f_avg_val += f
        optimizer.step()
    group_sparsity, param_norm, _ = optimizer.compute_group_sparsity_param_norm()
    norm_important, norm_redundant, num_grps_important, num_grps_redundant = optimizer.compute_norm_groups()
    accuracy1, accuracy5 = check_accuracy(model, testloader)
    f_avg_val = f_avg_val.cpu().item() / len(trainloader)
    
    print("Ep: {ep}, loss: {f:.2f}, norm_all:{param_norm:.2f}, grp_sparsity: {gs:.2f}, acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, num_grp_redund: {num_grps_redund}"\
         .format(ep=epoch, f=f_avg_val, param_norm=param_norm, gs=group_sparsity, acc1=accuracy1,\
         norm_import=norm_important, norm_redund=norm_redundant, num_grps_import=num_grps_important, num_grps_redund=num_grps_redundant
        ))

oto.construct_subnet(out_dir='/home/tianyi/otov2_auto_structured_pruning/cache')

full_model = torch.load(oto.full_group_sparse_model_path).cpu()
compressed_model = torch.load(oto.compressed_model_path).cpu()

full_output = full_model(dummy_input)
compressed_output = compressed_model(dummy_input)

import os
max_output_diff = torch.max(torch.abs(full_output - compressed_output))
print("Maximum output difference " + str(max_output_diff.item()))
full_model_size = os.stat(oto.full_group_sparse_model_path)
compressed_model_size = os.stat(oto.compressed_model_path)
print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")