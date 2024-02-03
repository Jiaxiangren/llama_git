from ast import arg
import copy
from http import client
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import SGD
from sklearn.metrics import f1_score, matthews_corrcoef
from statistics import mean
from torch import nn
import time
import math


logger = logging.getLogger(__name__)

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
    'wnli': 'wnli',
    'boolq': 'boolq',
    'mr': 'mr'
}


class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def set_seed(args):
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

def ratio_minus(w1, P, ratio=0):
    w = {}
    for key in w1.keys():
        if key in P:
            # print("key name:", key)
            w[key] = w1[key] - P[key] * ratio
    return w

def average_weights_plus(w, v, ratio=0):

    """
    Returns the average of the weights w and plus v.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = 0
    for i in range(0, len(w)):
        total += w[i][0]
    for key in w_avg.keys():
        w_avg[key] *= w[0][0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key] * w[i][0]
        w_avg[key] = ratio * torch.div(w_avg[key], total) + v[key]

    return w_avg


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = 0
    for i in range(0, len(w)):
        total += w[i][0]
    for key in w_avg.keys():
        # print("key name:", key)
        w_avg[key] *= w[0][0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key] * w[i][0]
        w_avg[key] = torch.div(w_avg[key], total)
    return w_avg

def evaluate(args, model, tokenizer):

    eval_task = args.task_name

    # fl_config.eval_batch_size = fl_config.per_gpu_eval_batch_size * max(1, args.n_gpu)

    results = {}

    eval_datasets = load_and_cache_data(args, tokenizer, data_type='dev')

    preds = []
    out_label_ids = None
    eval_loss = []
    nb_eval_steps = 0
    for i, eval_dataset in enumerate(eval_datasets):
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        losses = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.cuda() for t in batch)

            labels = batch[-1]

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1]
                }
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits
                loss = get_loss(args, logits, batch)

            eval_loss += loss[labels == i].cpu().detach().numpy().tolist()
            losses += loss.cpu().detach().numpy().tolist()

            if i == 0:
                if out_label_ids is None:
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        preds.append(losses)

    eval_loss = mean(eval_loss)
    preds = np.array(preds)
    preds = preds.transpose()

    preds = np.argmin(preds, axis=1)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))

    return eval_loss, result


def evaluate_llama(args, model, valid_loader):

    model.eval()
    total_loss = 0.
    start_time = time.time()

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}

            _input = data['input'].cuda()
            _target = data['target'].cuda()
            _msk = data['mask'].cuda()

            print(_input)
            exit()
            inputs = {
                "input_ids": _input,
                "attention_mask": _msk,
                "labels": _target
            }

            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss.mean() 
            
            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss.float())

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train(args, model, tokenizer, train_dataloader):


    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)     

    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_local_train_epochs 

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.warmup_steps > 0:
        num_warmup_steps = args.warmup_steps
    else:
        num_warmup_steps = args.warmup_rate * t_total

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level) 

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
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.rounds)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    local_step = 0
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_local_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )

    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        local_step = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, data in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            data = {key: value for key, value in data.items()}

            _input = data['input'].cuda()
            _target = data['target'].cuda()
            _msk = data['mask'].cuda()

            inputs = {
                "input_ids": _input,
                "attention_mask": _msk,
                "labels": _target
            }
            outputs = model(**inputs, return_dict=True)
            # logits = outputs.logits
            # loss = get_loss(args, logits, batch)
            loss = outputs.loss()

            # if args.method_type != 'discriminative':
            loss = loss.mean()
            # model.backward(loss)
            # model.step()
            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                local_step += 1

                # _, results = evaluate(args, model, tokenizer)
                            
        logging_loss = tr_loss
    
    state_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            # print(name)
            state_dict[name] = copy.deepcopy(p.data.detach().cpu())

    return state_dict, tr_loss / global_step

def get_loss(args, logits, batch_data):
    loss = None
    if args.prompt_type != 'v2':
        logits = logits[..., args.num_prompt_tokens:-1, :].contiguous()
    else:
        logits = logits[..., :-1, :].contiguous()
    labels = batch_data[0]
    labels = labels[..., 1:].contiguous()
    loss_mask = batch_data[2][..., 1:]

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]
    loss = loss.view(logits.size(0), logits.size(1)) * loss_mask.view(logits.size(0), logits.size(1))
    loss = torch.sum(loss, axis=1) / torch.sum(loss_mask, axis=1)   

    return loss