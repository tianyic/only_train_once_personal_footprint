# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import glob
import csv
import yaml
import time
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from optimizer.hspg import HSPG
from utils.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended, create_group_params_config)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', required=True, type=str)
    return parser.parse_args()

def set_seed(opt):
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    if opt['n_gpu'] > 0:
        torch.cuda.manual_seed_all(opt['seed'])

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(opt, train_dataset, model, tokenizer):
    """ Train the model """    
    print("Checkpoint directory: ", opt['checkpoint_dir'])
    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])

    setting = "bert_squad_training_" + opt['param_setting']
    csvname = 'results/%s.csv'%(setting)
    print('The csv file is %s'%csvname)
    csvfile = open(csvname, 'w', newline='')
    
    if opt['optimizer']['name'] == 'hspg':
        fieldnames = ['epoch', 'iter', 'F_value', 'f_value', 'omega_value', 'sparsity_group', 'sparsity_group_type_2', 'sparsity_group_type_3', \
            'exact', 'f1', 'train_time', 'step_size', 'lambda', 'eps', 'remarks']
    else:
        fieldnames = ['epoch', 'iter', 'f_value', 'exact', 'f1', 'train_time', 'step_size']    

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()
    csvfile.flush()

    train_batch_size = opt['train']['batch_size']
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    num_train_epochs = opt['train']['max_epoch']
    gradient_accumulation_steps = opt['train']['gradient_accumulation_steps']
    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    print(train_batch_size, num_train_epochs)


    if opt['optimizer']['name'] == 'hspg':
        optimizer_grouped_parameters = create_group_params_config(model, opt['optimizer']['epsilon'], opt['optimizer']['upper_group_sparsity'])
        optimizer = HSPG(optimizer_grouped_parameters, lr=opt['optimizer']['init_lr'], lmbda=opt['optimizer']['lambda'], momentum=opt['optimizer']['momentum'])
        print(optimizer.compute_group_sparsity_omega())
    elif opt['optimizer']['name'] == 'adamw':
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt['optimizer']['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=opt['optimizer']['init_lr'], eps=opt['optimizer']['epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=opt['lr_scheduler']['warmup_steps'], t_total=t_total)

    # Train!
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch")
    set_seed(opt)  # Added here for reproductibility (even between python 2 and 3)

    print(train_iterator)
    save_steps = opt['train']['save_ckpt_freq']

    do_half_space = False
    prev_group_sparsity = 0.0
    updated_epsilons = None
    stage = "sgd"
    for epoch in train_iterator:
        
        if opt['optimizer']['name'] == 'hspg':
            if epoch >= opt['optimizer']['n_p']:
                do_half_space = True
                stage = "half_space" if do_half_space else "sgd"

        for _ in range(opt['train']['train_times']):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            step_start_time = time.time()
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(opt['device']) for t in batch)
                inputs = {'input_ids':       batch[0],
                        'attention_mask':  batch[1], 
                        'token_type_ids':  batch[2],  
                        'start_positions': batch[3], 
                        'end_positions':   batch[4]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['optimizer']['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if opt['optimizer']['name'] == 'hspg':               
                        if do_half_space is False:
                            optimizer.sgd_step()
                        else:
                            optimizer.half_space_step()
                    else:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule

                    model.zero_grad()
                
                global_step += 1
                # adapt epsilon based on current group sparsity
                if opt['optimizer']['name'] == 'hspg' and do_half_space: 
                    if opt['optimizer']['adapt_epsilon'] is not None and opt['optimizer']['adapt_epsilon_freq'] is not None:
                        if global_step % opt['optimizer']['adapt_epsilon_freq'] == 0:
                            adapted, curr_group_sparsity, tmp_updated_epsilons = optimizer.adapt_epsilon(opt['optimizer']['adapt_epsilon'], opt['optimizer']['upper_group_sparsity'], prev_group_sparsity)
                            updated_epsilons = tmp_updated_epsilons if adapted else updated_epsilons
                            prev_group_sparsity = curr_group_sparsity

                if global_step == 1 or global_step % opt['train']['log_ckpt_freq'] == 0:
                    # Log metrics
                    results = None
                    if opt['train']['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        if global_step % (opt['train']['log_ckpt_freq'] * opt['train']['evaluate_ckpt_freq']) == 0:
                            results = evaluate(opt, model, tokenizer)
                    if global_step != 1:
                        logging_loss = tr_loss / float(opt['train']['log_ckpt_freq'])
                    else:
                        logging_loss = tr_loss
                        
                    train_time = time.time() - step_start_time
                    step_start_time = time.time()
                    if results is None:
                        f1, exact = "N/A", "N/A"
                    else:
                        f1, exact = results['f1'], results['exact']
                    if opt['optimizer']['name'] == 'hspg':    
                        n_zero_groups, n_groups, group_sparsities, overall_group_sparsity, omega = optimizer.compute_group_sparsity_omega()
                        psi = logging_loss + optimizer.param_groups[0]['lmbda'] * omega
                        logging_row = {'epoch': epoch, 'iter': step, 'F_value': psi, 'f_value': logging_loss, 'omega_value': omega, \
                            'sparsity_group': overall_group_sparsity, 'sparsity_group_type_2': group_sparsities[2], 'sparsity_group_type_3': group_sparsities[3], \
                            'exact': exact, 'f1': f1, 'train_time': train_time, 'step_size': optimizer.param_groups[0]['lr'], 'eps': updated_epsilons if updated_epsilons is not None else opt['optimizer']['epsilon'], \
                            'lambda': optimizer.param_groups[0]['lmbda'], 'remarks': '%s;'%(stage)}
                    else:
                        logging_row = {'epoch': epoch, 'iter': step, 'f_value': logging_loss, 'exact': exact, 'f1': f1, 'train_time': train_time, 'step_size': optimizer.param_groups[0]['lr']}

                    writer.writerow(logging_row)
                    csvfile.flush()

                    tr_loss = 0
                    
                    # Save model checkpoint
                    if global_step % (opt['train']['log_ckpt_freq'] * opt['train']['save_ckpt_freq']) == 0:
                        output_dir = os.path.join(opt['checkpoint_dir'], 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(opt, os.path.join(output_dir, 'training_opts.bin'))

                if opt['train']['max_steps'] > 0 and global_step > opt['train']['max_steps']:
                    epoch_iterator.close()
                    break

        # Save model checkpoint per epoch
        output_dir = os.path.join(opt['checkpoint_dir'], 'checkpoint-epoch-{}'.format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(opt, os.path.join(output_dir, 'training_opts.bin'))

        if opt['optimizer']['name'] == 'hspg':  
            if epoch in opt['optimizer']['decay_lambda_epochs']:
                for param_group in optimizer.param_groups:      
                    param_group['lmbda'] = 0.0 if param_group['lmbda'] <= 1e-6 else param_group['lmbda'] / float(10)

            if epoch in opt['optimizer']['decay_lr_epochs']:
                for param_group in optimizer.param_groups:      
                    param_group['lr'] /= float(10) # multi-step

        if opt['train']['max_steps'] > 0 and global_step > opt['train']['max_steps']:
            train_iterator.close()
            break
        
    return global_step, tr_loss / global_step


def evaluate(opt, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(opt, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(opt['checkpoint_dir']):
        os.makedirs(opt['checkpoint_dir'])

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_batch_size = opt['eval']['batch_size']
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(opt['device']) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(opt['checkpoint_dir'], "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(opt['checkpoint_dir'], "nbest_predictions_{}.json".format(prefix))
    if opt['version_2_with_negative']:
        output_null_log_odds_file = os.path.join(opt['checkpoint_dir'], "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
    
    write_predictions(examples, features, all_results, opt['eval']['n_best_size'],
                    opt['eval']['max_answer_length'], opt['do_lower_case'], output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, opt['eval']['verbose_logging'],
                    opt['version_2_with_negative'], 0.0)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=opt['eval_file'],
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    print(results)
    return results


def load_and_cache_examples(opt, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    print("Load data features from cache or dataset file")
    input_file = opt['eval_file'] if evaluate else opt['train_file']
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, opt['model_name_or_path'].split('/'))).pop(),
        str(opt['max_seq_length'])))
    
    if os.path.exists(cached_features_file) and not output_examples:
        features = torch.load(cached_features_file)
    else:        
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=opt['version_2_with_negative'])
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=opt['max_seq_length'],
                                                doc_stride=opt['doc_stride'],
                                                max_query_length=opt['max_query_length'],
                                                is_training=not evaluate)
        torch.save(features, cached_features_file)

    print("Convert to Tensors and build dataset")
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():

    args = ParseArgs()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt['name'] = os.path.basename(args.opt)[:-4]
    print('option:', opt)

    # Setup GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt['n_gpu'] = torch.cuda.device_count()
    opt['device'] = device

    # Set seed
    set_seed(opt)

    if opt['backend'] != "bert":
        raise("backend is not yet supported!")

    config_class, model_class, tokenizer_class = BertConfig, BertForQuestionAnswering, BertTokenizer
    config = config_class.from_pretrained(opt['model_name_or_path'])
    tokenizer = tokenizer_class.from_pretrained(opt['model_name_or_path'], do_lower_case=opt['do_lower_case'])
    model = model_class.from_pretrained(opt['model_name_or_path'], config=config)

    model.to(opt['device'])
    param_setting = "_" + opt['optimizer']['name'] + "_" + args.opt.split("/")[-1]
    
    opt['checkpoint_dir'] = os.path.join(opt['checkpoint_dir'] + param_setting)
    opt['param_setting'] = param_setting
    
    if opt['ckpt_initial'] is not None:
        pretrain_path = opt['ckpt_initial']
        pretrain_state_dict = torch.load(pretrain_path, map_location=device)
        if opt['load_embedding_only']:
            for i, (key, pretrain_key) in enumerate(zip(model.state_dict(), pretrain_state_dict)):
                param = model.state_dict()[key]
                pretrain_param = pretrain_state_dict[pretrain_key]
                if "embeddings" in key and 'embeddings' in pretrain_key:
                    param.data.copy_(pretrain_param)
                    # print(i, key, pretrain_key, pretrain_state_dict[key].shape)
        else:
            model.load_state_dict(torch.load(pretrain_path, map_location=device))
        # result = evaluate(opt, model, tokenizer)
    
    # Training
    print("Do Training...")
    if opt['do_train']:
        print("Load Training dataset")
        train_dataset = load_and_cache_examples(opt, tokenizer, evaluate=False, output_examples=False)
        print("Start training")
        global_step, tr_loss = train(opt, train_dataset, model, tokenizer)

    # Evaluation 
    results = {}
    if opt['do_eval']:
        # Evaluate
        result = evaluate(opt, model, tokenizer, prefix=global_step)

        result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
        results.update(result)

    return results


if __name__ == "__main__":
    main()
