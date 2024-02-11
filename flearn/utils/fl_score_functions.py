from ast import arg
import json
from lib2to3.pgen2 import token
import logging
import os
import random
import sys
import fitlog
import pickle

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import transformers
from torch.utils.data import DataLoader

from data_loader import PromptSortDataset
from scipy import linalg
import math

logger = logging.getLogger(__name__)
START = 0
END = 2
MASK = 50265

task_mappings = {
    'sst-2': 'sst-2',
    'cola': 'cola',
    'mnli': 'mnli',
    'mnli-mm': 'mnli-mm',
    'qqp': 'qqp',
    'qnli': 'qnli',
    'rte': 'rte',
    'mrpc': 'mrpc',
    'mpqa': 'sst-2',
    'mr': 'sst-2',
    'subj': 'sst-2',
    'trec': 'sst-2',
    'snli': 'qnli',
}

def get_metric_key(task_name):
    if task_name == "cola":
        return "mcc"
    elif task_name == "sst-2":
        return "acc"
    elif task_name == "mrpc":
        return "acc_and_f1"
    elif task_name == "sts-b":
        return "corr"
    elif task_name == "qqp":
        return "acc_and_f1"
    elif task_name == "mnli":
        return "mnli/acc"
    elif task_name == "mnli-mm":
        return "mnli-mm/acc"
    elif task_name == "qnli":
        return "acc"
    elif task_name == "rte":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "hans":
        return "acc"
    elif task_name == "mpqa":
        return "acc"
    elif task_name == "mr":
        return "acc"
    elif task_name == "subj":
        return "acc"
    elif task_name == "trec":
        return "acc"
    elif task_name == "snli":
        return "acc"
    elif task_name == "boolq":
        return "acc"
    else:
        raise KeyError(task_name)

        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate_and_sort_ours(args, train_dataset, model, client_index, col_func):

    model_config, fl_config = args

    model_config.train_batch_size = fl_config.train_batch_size * max(1, fl_config.n_gpu)
    train_sampler = RandomSampler(train_dataset) if fl_config.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs



    # Train!
    logger.info("***** Evaluating Data Samples *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", fl_config.num_local_train_epochs)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     fl_config.train_batch_size
    #     * fl_config.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if fl_config.num_local_train_epochs != -1 else 1),
    # )
    logger.info("  Gradient Accumulation steps = %d", fl_config.gradient_accumulation_steps)


    # set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name)
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    sorted_tuple = []

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        fl_config.logging_steps = len(epoch_iterator)
        # args.logging_steps = 1
        # Skip past any already trained steps if resuming training

        model.eval()
        batch = tuple(t.to(model_config.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[-1],
        }
        inputs["token_type_ids"] = batch[2]
        inputs["mask_pos"] = batch[-2]
        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[2]
        print("length of tuple:", len(hidden_states))
        total_sum = 0
        for x in hidden_states[1:]:
            if x.size()[0] != fl_config.train_batch_size:
                continue
            x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
            # x_cov = np.cov(x.detach().cpu().T)
            x_cov = torch.cov(x.T)
            
            w = torch.view_as_real(torch.linalg.eigvals(x_cov))
            # w = linalg.eigvals(x_cov)
            # eig_sum = np.sum(w).astype(np.float32)
            eig_sum = torch.sum(w)
            # total_sum += np.log(eig_sum)
            total_sum += np.log(eig_sum.item())
        
        print(total_sum)
        batch = tuple(t.detach().cpu() for t in batch)
        sorted_tuple.append((-total_sum, batch))
    
    sorted_tuple = sorted(sorted_tuple, key=lambda x: x[0])
    # print(sorted_tuple)
    return sorted_tuple

def evaluate_and_sort_ours_fisher(args, train_dataset, model, client_index, col_func):

    model_config, fl_config = args

    model_config.train_batch_size = fl_config.train_batch_size * max(1, fl_config.n_gpu)
    train_sampler = RandomSampler(train_dataset) if fl_config.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs



    # Train!
    logger.info("***** Evaluating Data Samples *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", fl_config.num_local_train_epochs)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     fl_config.train_batch_size
    #     * fl_config.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if fl_config.num_local_train_epochs != -1 else 1),
    # )
    logger.info("  Gradient Accumulation steps = %d", fl_config.gradient_accumulation_steps)


    # set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name)
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    sorted_tuple = []

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        fl_config.logging_steps = len(epoch_iterator)
        # args.logging_steps = 1
        # Skip past any already trained steps if resuming training
        model.train()
        batch = tuple(t.to(model_config.device) for t in batch)
        input_ids = batch[0]
        batch_num = input_ids.size()[0]
        est_fisher_info = {}

        for i in range(batch_num):
            # print("estimating for sample:", i)
            inputs = {
                "input_ids": batch[0][i].unsqueeze(0),
                "attention_mask": batch[1][i].unsqueeze(0),
                "labels": batch[-1][i].unsqueeze(0),
            }
            inputs["token_type_ids"] = batch[2][i].unsqueeze(0)
            inputs["mask_pos"] = batch[-2][i].unsqueeze(0)
            outputs = model(**inputs)
            loss = outputs[0]
            model.zero_grad()
            loss.backward()
            # calculate the approximate fisher information matrix
            for n, p in model.named_parameters(): 
                if p.requires_grad: 
                    # n = n.replace('.', '__') 
                    if p.grad is not None: 
                        if n not in est_fisher_info:
                            est_fisher_info[n] = 0
                        est_fisher_info[n] += torch.sum(p.grad.detach() ** 2)
        est_fisher_info = {n: p/batch_num for n, p in est_fisher_info.items()}
        score = 0

        # calculate the trace
        for key, value in est_fisher_info.items():
            score += sum(torch.flatten(value))
        # score = sum(est_fisher_info.values())
        batch = tuple(t.detach().cpu() for t in batch)
        sorted_tuple.append((score, batch))
    
    sorted_tuple = sorted(sorted_tuple, key=lambda x: x[0])
    # print(sorted_tuple)
    return sorted_tuple

def evaluate_and_sort_voc_freq(unigrad_dict, total, train_dataset):

    from collections import Counter

    for sentence in train_dataset:
        input_ids = sentence[0]
        
        for id in input_ids:
            if id not in unigrad_dict:
                unigrad_dict[id] = 0
            unigrad_dict[id] += 1
            total += 1
    
    return unigrad_dict, total

def evaluate_and_sort_voc(args, model_args, train_dataset, model, tokenizer, global_dict, total, save=False):

    from collections import Counter
    
    total_score = 0
    uncount_index_list = [START, END, MASK]
    score_dataset = []
    #initialize the uni-gram dict

    for sentence in train_dataset:
        input_ids = sentence[0]
        
        score = 0
        for id in input_ids:
            if id in uncount_index_list:
                continue

            score += -math.log(global_dict[id]/total)

        score_dataset.append((score, sentence))
        total_score += score

    client_score = total_score / len(train_dataset)
    score_dataset = PromptSortDataset(args, model_args.task_name, tokenizer, score_dataset, data_type='train')
    
    return score_dataset, client_score


def evaluate_and_sort_seqreo(args, model_args, train_dataset, model, tokenizer, save=False):
    
    score_dataset = []
    total_score = 0

    for sentence in train_dataset:

        input_ids = sentence[0]
        length = len(input_ids)

        score_dataset.append((length, sentence))

        total_score += length
    
    client_score = total_score / len(train_dataset)
    score_dataset = PromptSortDataset(args, model_args.task_name, tokenizer, score_dataset, data_type='train')
    if save:
        file_name = './sorted_dataset/{}_{}_dataset.pkl'.format(args.sort_type, model_args.task_name)
        with open(file_name, "wb") as f:
            pickle.dump(score_dataset, f)

    
    
    return score_dataset, client_score


def evaluate_and_sort_loss(args, model_args, train_dataset, model, tokenizer):

    score_dataset = []

    for sentence in tqdm(train_dataset):

        model.eval()
        batch = tuple(t for t in sentence)
        inputs = {
            "input_ids": torch.tensor(batch[0], dtype=torch.long).view(1,-1).to(model_args.device),
            "attention_mask": torch.tensor(batch[1], dtype=torch.long).view(1,-1).to(model_args.device),
            "labels": torch.tensor(batch[-1], dtype=torch.long).view(1,-1).to(model_args.device),
        }
        inputs["token_type_ids"] = torch.tensor(batch[2], dtype=torch.long).view(1,-1).to(model_args.device)
        inputs["mask_pos"] = torch.tensor(batch[-2], dtype=torch.long).view(1,-1).to(model_args.device)
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        sentence = tuple(t for t in batch)
        score_dataset.append((loss.item(), sentence))
    
    score_dataset = PromptSortDataset(args, model_args.task_name, tokenizer, score_dataset, data_type='train')
    
    return score_dataset, 1


def evaluate_and_sort_ours_voc(args, train_dataset, model, tokenizer, col_func):

    model_config, fl_config = args

    
    # model_config.train_batch_size = fl_config.train_batch_size * max(1, fl_config.n_gpu)

    train_dataset = evaluate_and_sort_voc(fl_config, model_config, train_dataset, model, tokenizer)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs



    # Train!
    logger.info("***** Evaluating Data Samples *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", fl_config.num_local_train_epochs)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     fl_config.train_batch_size
    #     * fl_config.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if fl_config.num_local_train_epochs != -1 else 1),
    # )
    logger.info("  Gradient Accumulation steps = %d", fl_config.gradient_accumulation_steps)


    # set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name)
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    sorted_tuple = []
    total_score = 0

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        fl_config.logging_steps = len(epoch_iterator)
        # args.logging_steps = 1
        # Skip past any already trained steps if resuming training

        model.eval()
        batch = tuple(t.to(model_config.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[-1],
        }
        inputs["token_type_ids"] = batch[2]
        inputs["mask_pos"] = batch[-2]
        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[2]
        print("length of tuple:", len(hidden_states))
        total_sum = 0
        for x in hidden_states[1:]:
            if x.size()[0] != fl_config.train_batch_size:
                continue
            x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
            # x_cov = np.cov(x.detach().cpu().T)
            x_cov = torch.cov(x.T)
            
            w = torch.view_as_real(torch.linalg.eigvals(x_cov))
            # w = linalg.eigvals(x_cov)
            # eig_sum = np.sum(w).astype(np.float32)
            eig_sum = torch.sum(w)
            # total_sum += np.log(eig_sum)
            total_sum += np.log(eig_sum.item())
        
        print(total_sum)
        batch = tuple(t.detach().cpu() for t in batch)
        sorted_tuple.append((-total_sum, batch))
        total_score = total_score - total_sum
    
    client_score = total_score / len(train_dataloader)
    sorted_tuple = sorted(sorted_tuple, key=lambda x: x[0])
    # print(sorted_tuple)
    return sorted_tuple


def evaluate_and_sort_ours_seq(training_args, args, train_dataset, model, tokenizer):


    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    
    train_dataset = evaluate_and_sort_seqreo(training_args, args, train_dataset, model, tokenizer, save=False)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Evaluating Data Samples *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)


    set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(args.task_name)
    if args.task_name == 'mnli':
        metric_key = 'avg_acc'

    sorted_tuple = []

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        args.logging_steps = len(epoch_iterator)
        # args.logging_steps = 1
        # Skip past any already trained steps if resuming training

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[-1],
        }
        inputs["token_type_ids"] = batch[2]
        inputs["mask_pos"] = batch[-2]
        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[2]
        print("length of tuple:", len(hidden_states))
        total_sum = 0
        for x in hidden_states[1:]:
            x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
            x_cov = np.cov(x.detach().cpu().T)
            w = linalg.eigvals(x_cov)
            eig_sum = np.sum(w).astype(np.float32)
            total_sum += np.log(eig_sum)
        
        print(total_sum)
        batch = tuple(t.detach().cpu() for t in batch)
        sorted_tuple.append((-total_sum, batch))
        

    file_name = './sorted_dataset/{}_{}_dataset.pkl'.format(training_args.sort_type, args.task_name)
    with open(file_name, "wb") as f:
        pickle.dump(sorted_tuple, f)
    

    return sorted_tuple


def evaluate_and_sort_ours_last(training_args, args, train_dataset, model, tokenizer):


    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Evaluating Data Samples *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)


    set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(args.task_name)
    if args.task_name == 'mnli':
        metric_key = 'avg_acc'

    sorted_tuple = []

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        args.logging_steps = len(epoch_iterator)
        # args.logging_steps = 1
        # Skip past any already trained steps if resuming training

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[-1],
        }
        inputs["token_type_ids"] = batch[2]
        inputs["mask_pos"] = batch[-2]
        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[2]
        print("length of tuple:", len(hidden_states))
        total_sum = 0
        x = hidden_states[-1]
        x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
        x_cov = np.cov(x.detach().cpu().T)
        w = linalg.eigvals(x_cov)
        eig_sum = np.sum(w).astype(np.float32)
        total_sum += np.log(eig_sum)
        
        print(total_sum)
        batch = tuple(t.detach().cpu() for t in batch)
        sorted_tuple.append((-total_sum, batch))

    file_name = './sorted_dataset/{}_{}_dataset.pkl'.format(training_args.sort_type, args.task_name)
    with open(file_name, "wb") as f:
        pickle.dump(sorted_tuple, f)
    

    return sorted_tuple


def evaluate_and_sort_ours_v2(training_args, args, train_dataset, model, tokenizer):


    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir) 

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Evaluating Data Samples *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)


    set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(args.task_name)
    if args.task_name == 'mnli':
        metric_key = 'avg_acc'

    sorted_tuple = []

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        args.logging_steps = len(epoch_iterator)
        # args.logging_steps = 1
        # Skip past any already trained steps if resuming training

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[-1],
        }
        inputs["token_type_ids"] = batch[2]
        inputs["mask_pos"] = batch[-2]
        outputs = model(**inputs)
        loss = outputs[0]
        hidden_states = outputs[2]
        print("length of tuple:", len(hidden_states))
        total_sum = 0
        for x in hidden_states[1:]:
            x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
            x_cov_1 = abs(x.detach().cpu()) * abs(x.detach().cpu().T)
            for i in range(x_cov_1.shape[0]):
                x_cov_1[i][i] = x_cov_1[i][i] * 2
            x_cov = x.detach().cpu() @ x.detach().cpu().T
            x_cov_1 = x_cov_1 - x_cov
            
            w = np.linalg.det(x_cov_1).astype(np.float32)
            # w = linalg.eigvals(x_cov)
            # eig_sum = np.sum(w).astype(np.float32)
            total_sum += np.log(w)
        
        print(total_sum)
        batch = tuple(t.detach().cpu() for t in batch)
        sorted_tuple.append((total_sum, batch))

    file_name = './sorted_dataset/{}_{}_dataset.pkl'.format(training_args.sort_type, args.task_name)
    with open(file_name, "wb") as f:
        pickle.dump(sorted_tuple, f)
    

    return sorted_tuple
    

    
    
    
    
        
    
