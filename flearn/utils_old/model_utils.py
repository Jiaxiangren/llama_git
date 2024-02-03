import copy
from http import client
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import glue_compute_metrics
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import SGD


from flearn.utils.process_data import PromptDataset
from Attempt.data.process import processors
from flearn.optim.fadamw import fAdamW
from flearn.optim.fsgdm import Fsgdm

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


def update_client_c(global_c, client_c, delta, local_steps):
    #client_c[k] = client_c[k] - global_c[k] + (original_params[k].cpu().detach() - new_params[k].cpu().detach()) / (local_steps * fl_config.learning_rate)
    res = {}
    for k in global_c:
        res[k] = client_c[k] - global_c[k] + delta[k] / local_steps
    
    return res
    

def calculate_big_G(p, m, beta):
    res = {}
    for key in p:
        if not m:
            res[key] = p[key] / (1 - beta)
        else:
            res[key] = (p[key] - beta * m[key]) / (1 - beta)
    return res


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

    model_config, prompt_config, fl_config = args[0], args[1], args[2]

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if model_config.task_name == "mnli" else (model_config.task_name,)
    eval_outputs_dirs = (model_config.output_dir, model_config.output_dir + "-MM") if model_config.task_name == "mnli" else (model_config.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task.lower()]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        fl_config.eval_batch_size = fl_config.eval_batch_size * max(1, fl_config.n_gpu)

        if os.path.exists(os.path.join(model_config.data_dir, 'saved_testset.pkl')):
            eval_dataset = torch.load(os.path.join(model_config.data_dir, 'saved_testset.pkl'))
        else:
            eval_dataset = PromptDataset(model_config, eval_task, tokenizer, data_type='dev')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=fl_config.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and model_config.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if fl_config.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        # logger.info("***** Running evaluation *****")
        # logger.info("  Num examples = %d", len(eval_dataset))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(model_config.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[-1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[-2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                logits = logits[:, label_ids]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = batch_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = np.append(out_label_ids, batch_labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if model_config.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif model_config.output_mode == "regression":
            preds = np.squeeze(preds)
        result = glue_compute_metrics(task_mappings[eval_task.lower()], preds, out_label_ids)
        results.update(result)
        

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

    return eval_loss, results


def train_scaffold(args, train_dataloader, model, v={}, global_c=None, client_c=None):

    model_config, fl_config = args[0], args[1]

    # if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir)

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]  

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

    if fl_config.local_optimizer == 'fAdamw':
        optimizer = fAdamW(model, v=v, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)
    elif fl_config.local_optimizer == 'Adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)
    elif fl_config.local_optimizer == "fSgdm":
        optimizer = Fsgdm(model, lr=fl_config.learning_rate, momentum=0.9, v=v)



    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader)*fl_config.train_batch_size)
    logger.info("  Num Epochs = %d", fl_config.num_local_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", fl_config.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        fl_config.train_batch_size
        * fl_config.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if fl_config.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", fl_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(fl_config.num_local_train_epochs),
        desc="Epoch",
        disable=fl_config.local_rank not in [-1, 0],
    )

    set_seed(model_config)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name.lower())
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    original_params = model.get_copy_of_trainable_weights()
    original_client_c = copy.deepcopy(client_c)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
            # args.logging_steps = 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            inputs["mask_pos"] = batch[-2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()    

            tr_loss += loss.item()

            if not fl_config.batch_grad:
                if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                    if fl_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
        
        if fl_config.batch_grad:
            tr_loss = tr_loss / len(epoch_iterator)
            for p in model.parameters():
                if p.requires_grad==True:
                    p.grad.data = p.grad.data / len(epoch_iterator)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
    
    state_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            # print(name)
            state_dict[name] = copy.deepcopy(p.data)
    
    # update the local c and return the delta c
    
    
    
    if fl_config.local_optimizer == 'fAdamw' or fl_config.local_optimizer == "fSgdm":
        # p = p + c_global - c_i
        for key in optimizer.P:
            optimizer.P[key] += (global_c[key].to(model_config.device) - client_c[key].to(model_config.device))

        # update the c_i and delta c
        new_params = model.get_copy_of_trainable_weights()
        delta_c = {}
        for k in original_params:
            local_steps = fl_config.num_local_train_epochs * len(epoch_iterator)
            client_c[k] = client_c[k] - global_c[k] + (original_params[k].cpu().detach() - new_params[k].cpu().detach()) / (local_steps * fl_config.learning_rate)
            delta_c[k] = client_c[k] - original_client_c[k].cpu().detach()
        return state_dict, tr_loss/global_step, optimizer.v, optimizer.P, delta_c, client_c
    return state_dict, tr_loss / global_step


def train(args, train_dataloader, model, v={}):

    model_config, fl_config = args[0], args[1]

    # if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir)

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]  

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

    if fl_config.local_optimizer == 'fAdamw':
        optimizer = fAdamW(model, v=v, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)
    elif fl_config.local_optimizer == 'Adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)
    elif fl_config.local_optimizer == "fSgdm":
        optimizer = Fsgdm(model, lr=fl_config.learning_rate, momentum=0.9, v=v)



    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader)*fl_config.train_batch_size)
    logger.info("  Num Epochs = %d", fl_config.num_local_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", fl_config.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        fl_config.train_batch_size
        * fl_config.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if fl_config.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", fl_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(fl_config.num_local_train_epochs),
        desc="Epoch",
        disable=fl_config.local_rank not in [-1, 0],
    )

    set_seed(model_config)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name.lower())
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
            # args.logging_steps = 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            inputs["mask_pos"] = batch[-2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()    

            tr_loss += loss.item()

            if not fl_config.batch_grad:
                if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                    if fl_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
        
        if fl_config.batch_grad:
            tr_loss = tr_loss / len(epoch_iterator)
            for p in model.parameters():
                if p.requires_grad==True:
                    p.grad.data = p.grad.data / len(epoch_iterator)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
    
    state_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            # print(name)
            state_dict[name] = copy.deepcopy(p.data)
    
    if fl_config.local_optimizer == 'fAdamw' or fl_config.local_optimizer == "fSgdm":
        return state_dict, tr_loss/global_step, optimizer.v, optimizer.P
    return state_dict, tr_loss / global_step


def train_simple(args, train_dataloader, model, v={}):

    model_config, fl_config = args[0], args[1]

    # if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir)

    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total
    if fl_config.local_optimizer == "Sgd":
        optimizer = SGD(model.parameters(), lr = fl_config.learning_rate)
    elif fl_config.local_optimizer == "Sgdm":
        optimizer = SGD(model.parameters(), lr = fl_config.learning_rate, momentum=0.9)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)



    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader)*fl_config.train_batch_size)
    logger.info("  Num Epochs = %d", fl_config.num_local_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", fl_config.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        fl_config.train_batch_size
        * fl_config.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if fl_config.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", fl_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(fl_config.num_local_train_epochs),
        desc="Epoch",
        disable=fl_config.local_rank not in [-1, 0],
    )

    set_seed(model_config)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name.lower())
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
            # args.logging_steps = 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            inputs["mask_pos"] = batch[-2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()    

            tr_loss += loss.item()

            if not fl_config.batch_grad:
                if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                    if fl_config.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
        
    
    state_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            # print(name)
            state_dict[name] = copy.deepcopy(p.data)
    
    local_steps = fl_config.num_local_train_epochs * len(epoch_iterator)
    
    return state_dict, tr_loss / global_step, local_steps




def predict(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_list = processor.get_labels()
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = 32
        eval_dataset = PromptDataset(args, eval_task, tokenizer, data_type='test')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None

        for batch in tqdm(eval_dataloader, desc="Infering"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[3]
                outputs = model(**inputs)
                logits = outputs[0]
                logits = logits[:, label_ids]

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        output_infer_file = os.path.join(
            eval_output_dir, 
            "{}_{}_{}_{}_{}_{}_{}.tsv".format(
                eval_task, 
                args.generator_type,
                args.add_prompt_layer, 
                args.num_prompt_tokens, 
                args.proj_down_size,
                args.per_gpu_train_batch_size,
                args.learning_rate,
                args.warmup_rate,
            )
        )
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "prediction"])
            for i, pred in enumerate(preds):
                if args.output_mode == "classification":
                    prediction = label_list[pred]
                elif args.output_mode == "regression":
                    prediction = str(pred)
                writer.writerow([i, prediction])