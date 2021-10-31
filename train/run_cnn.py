import time
import os
import sys
import csv
import yaml
import random
import torch
import numpy as np
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from backend import Model
from data.datasets import Dataset
from utils.utils import compute_func_values, check_accuracy
from optimizer import HSPG, ProxSG


def adjust_learning_rate(optimizer, epoch, decays):
    if epoch in decays:
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] / 10.0
    print('lr:', optimizer.param_groups[0]['lr'])

def adjust_lambda(optimizer, epoch, decays):
    if epoch in decays:
        for group in optimizer.param_groups:
            group['lmbda'] = 0.0 if group['lmbda'] <= 1e-6 else group['lmbda'] / 10.0
    print('lmbda:', optimizer.param_groups[0]['lmbda'])
 
def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', required=True, type=str)
    return parser.parse_args()

def set_seed(opt):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

def train(opt, trainloader, testloader, model):

    print("Checkpoint directory: ", opt['checkpoint_dir'])
    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])

    setting = "%s_%s_training%s"%(opt['backend'], opt['dataset_name'], opt['param_setting'])
    os.makedirs('results', exist_ok=True)
    csvname = 'results/%s.csv'%(setting)
    print('The csv file is %s'%csvname)
    csvfile = open(csvname, 'w', newline='')


    fieldnames = ['epoch', 'F_value', 'f_value', 'omega_value', 'sparsity_group', 'validation_acc1', 'validation_acc5', 'train_time', 'step_size', 'lambda', 'remarks']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()
    csvfile.flush()


    model.create_zig_params(opt['optimizer'])
    if opt['optimizer']['name'] == 'hspg':
        optimizer = HSPG(model.optimizer_grouped_parameters, \
            lr=opt['optimizer']['init_lr'], \
            lmbda=opt['optimizer']['lambda'], \
            momentum=opt['optimizer']['momentum'])
    else:
        raise ValueError

    criterion = torch.nn.CrossEntropyLoss()

    alg_start_time = time.time()

    epoch = 0

    while True:
        adjust_learning_rate(optimizer, epoch, opt['optimizer']['decays'])
        if 'decays_lambda' in opt['optimizer']:
            adjust_lambda(optimizer, epoch, opt['optimizer']['decays_lambda'])

        epoch_start_time = time.time()
        print("epoch {}".format(epoch), end = '...')
        
        if epoch >= opt['train']['max_epoch']:
            break

        for index, (X, y) in enumerate(trainloader):
            X = X.to(opt['device'])
            y = y.to(opt['device'])

            print(X.shape, y.shape)
            exit()
            y_pred = model.forward(X)
            f = criterion(y_pred, y)
            optimizer.zero_grad()
            f.backward()
            if 'max_grad_norm' in opt['optimizer']:
                if opt['optimizer']['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt['optimizer']['max_grad_norm'])
            if epoch < opt['optimizer']['n_p']:
                optimizer.sgd_step()
            else:
                optimizer.half_space_step()
        epoch += 1

        train_time = time.time() - epoch_start_time
        _, _, _, sparsity_group, omega = optimizer.compute_group_sparsity_omega()
        F, f = compute_func_values(trainloader, model, criterion, optimizer.param_groups[0]['lmbda'], omega)
        accuracy1, accuracy5 = evaluate(model, testloader)
        writer.writerow({'epoch': epoch, 'F_value': F, 'f_value': f, 'omega_value': omega, \
            'sparsity_group': sparsity_group,\
            'validation_acc1': accuracy1, 'validation_acc5':accuracy5, 'train_time': train_time, 'step_size':optimizer.param_groups[0]['lr'], 'lambda':optimizer.param_groups[0]['lmbda'], })
        csvfile.flush()
        print("Epoch time: {:2f}seconds".format(train_time), end='...')

        if epoch % 20 == 0:
            torch.save(model.state_dict(), 'checkpoints/' + setting+'.pt')

    alg_time = time.time() - alg_start_time
    writer.writerow({'train_time': alg_time / epoch})
    torch.save(model.state_dict(), 'checkpoints/' + setting+'.pt')
    csvfile.close()

def evaluate(model, testloader):
    accuracy1, accuracy5 = check_accuracy(model, testloader)
    return accuracy1, accuracy5
    

def main():
    
    args = ParseArgs()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt['name'] = os.path.basename(args.opt)[:-4]
    print('option:', opt)

    # Setup GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt['device'] = device

    # Set seed
    set_seed(opt)

    model = Model(backend=opt['backend'], device=opt['device'])
    model.to(opt['device'])
    param_setting = "_" + opt['optimizer']['name'] + "_seed" + str(opt['seed']) + "_" + opt['name']
    opt['checkpoint_dir'] = os.path.join(opt['checkpoint_dir'] + param_setting)
    opt['param_setting'] = param_setting
    print('param_setting:', param_setting)


    print("Load dataset")
    trainloader, testloader = Dataset(opt['dataset_name'], batch_size=opt['train']['batch_size'])

    # Training
    if opt['do_train']:
        print("Start training")
        train(opt, trainloader, testloader, model)

    if opt['do_eval']:
        print("Start evaluating")
        accuracy1, accuracy5 = evaluate(model, testloader)
        print("Acc 1: ", accuracy1, ", Acc 5: ", accuracy5)


if __name__ == "__main__":
    main()
