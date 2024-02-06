import copy
from http import client
from importlib.metadata import requires
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import glue_compute_metrics
from tqdm import tqdm, trange

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import torch.nn.functional as F


from flearn.utils.process_data import PromptDataset
from data.process import processors
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

tasks_num_labels = {
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "mpqa": 2,
    "mr": 2,
    "subj": 2,
    "trec": 6,
    "cola": 2,
    "wnli": 2,
    "boolq": 2
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


# def average_weights_freeze(w):
#     """
#     Returns the average of the weights.
#     """
#     w_avg = copy.deepcopy(w[0][1])
#     total = {}
#     for i in range(0, len(w)):
#         for key in w_avg.keys():
#             total += w[i][0]
#     for key in w_avg.keys():
#         # print("key name:", key)
#         w_avg[key] *= w[0][0]
#         for i in range(1, len(w)):
#             w_avg[key] += w[i][1][key] * w[i][0]
#         w_avg[key] = torch.div(w_avg[key], total)
#     return w_avg

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
            eval_key = key
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

    return eval_loss, results[eval_key]


def evaluate_personalized(args, model, test_loaders, tokenizer, client_weights, transfer_parameter_names):

    model_config = args
    acc_list = []
    eval_loss_list = []
    eval_task = model_config.task_name.lower()
    processor = processors[eval_task]()
    label_ids = []
    label_map = processor.get_label_map()
    for k, v in label_map.items():
        label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_ids.append(label_id[0])
    
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    transfer_weights = model.get_copy_of_transfer_weights(transfer_parameter_names)
    for client_index in range(model_config.num_clients):

        test_loader = test_loaders[client_index]
        model.train()
        model.update_trainable_weights_from_dict(copy.deepcopy(client_weights[client_index]))
        model.train()
        model.update_transfer_weights_from_dict(copy.deepcopy(transfer_weights))


        for batch in tqdm(test_loader, desc="Evaluating"):
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
    # results.update(result)
    
    # logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        eval_key = key
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))
        
    acc_list.append(result[eval_key])
    eval_loss_list.append(eval_loss)
    
    # print(acc_list)

    return eval_loss, result[eval_key]

def evaluate_personalized_freeze(args, model, test_loaders, tokenizer, client_weights, global_weights, clients_transfer_parameter_names):

    model_config = args
    acc_list = []
    eval_loss_list = []
    eval_task = model_config.task_name.lower()
    processor = processors[eval_task]()
    label_ids = []
    label_map = processor.get_label_map()
    for k, v in label_map.items():
        label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
        assert len(label_id) == 1
        label_ids.append(label_id[0])
    
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    
    for client_index in range(model_config.num_clients):
        
        model.train()
        model.update_trainable_weights_from_dict(global_weights)
        transfer_weights = model.get_copy_of_transfer_weights(clients_transfer_parameter_names[client_index])
        test_loader = test_loaders[client_index]
        model.train()
        model.update_trainable_weights_from_dict(copy.deepcopy(client_weights[client_index]))
        model.train()
        model.update_transfer_weights_from_dict(copy.deepcopy(transfer_weights))


        for batch in tqdm(test_loader, desc="Evaluating"):
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
    # results.update(result)
    
    # logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        eval_key = key
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))
        
    acc_list.append(result[eval_key])
    eval_loss_list.append(eval_loss)
    
    # print(acc_list)

    return eval_loss, result[eval_key]



def evaluate_personalized_verify(args, model, test_loaders, tokenizer, client_weights, transfer_parameter_names):

    model_config = args
    acc_list = []
    eval_loss_list = []
    
    
    
    transfer_weights = model.get_copy_of_transfer_weights(transfer_parameter_names)
    for client_index in range(model_config.num_clients):
        print(client_index)
        eval_task = model_config.task_name.lower()
        processor = processors[eval_task]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None

        test_loader = test_loaders[client_index]
        model.train()
        model.update_trainable_weights_from_dict(copy.deepcopy(client_weights[client_index]))
        if client_index in [0,83]:
            print("###############evaluate the weights for client: {}##################".format(client_index))
            for name, p in model.named_parameters():
                if name == 'roberta.encoder.layer.0.attention.self.query.lora_A':
                    print(p)

        model.eval()
        for batch in tqdm(test_loader, desc="Evaluating"):
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
        # results.update(result)
        
        # logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            eval_key = key
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))
        
        acc_list.append(result[eval_key])
        eval_loss_list.append(eval_loss)
    
    # print(acc_list)

    return eval_loss, result[eval_key]
    
    # print(acc_list)




def train(args, train_dataloader, model, col_func):

    model_config, fl_config = args[0], args[1]

    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]  

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

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

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            print("loss:", loss)      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        print(name, p.grad)
                exit()
                optimizer.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_plgu(args, train_dataloader, model, col_func):

    model_config, fl_config = args[0], args[1]

    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

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


    # evaluate the noise
    for _ in range(1):
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

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
    for _, params in model.named_parameters():
        if params.requires_grad:
            grad_norm = torch.norm(params.grad, p=2) + 1e-7
            # print("grad norm:", grad_norm)
            scale = 0.05 / grad_norm
            with torch.no_grad():
                epsilion = params.grad.data * scale
            # print("epsilion:", epsilion)
            with torch.no_grad():
                params.add_(epsilion)
    
    
    # adv_embeddings = embeddings + epsilion

    model.zero_grad()
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

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _

def train_fedalt(args, train_dataloader, model, transfer_parameter_names):

    model_config, fl_config = args[0], args[1]


    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in transfer_parameter_names],
            "weight_decay": fl_config.weight_decay,
        }
    ]


    optimizer_grouped_parameters_2 = [
        {
            "params": [p for n, p in model.named_parameters() if n not in transfer_parameter_names and p.requires_grad],
            "weight_decay": fl_config.weight_decay,
        }
    ]  

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

    optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)
    optimizer_2 = AdamW(optimizer_grouped_parameters_2, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)



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


            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
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
            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer_2.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_se(args, train_dataloader, model, col_func):

    model_config, fl_config = args[0], args[1]

    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    

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

            # if the loss above the thredshold, use the smoothed label softmax
            if loss > 2:
                # pre_logits = outputs[1].view(fl_config.train_batch_size, -1)
                # log_probs = F.log_softmax(pre_logits, dim=1)
                # probs = F.softmax(pre_logits, dim=1)
                # batch_size = batch[-1].size()
                # one_hot = torch.zeros(batch_size, tasks_num_labels[fl_config.task_name])
                # one_hot.scatter_(1, batch[-1].view(-1, 1).long(), 1)
                # soft_labels = (probs + one_hot) / 2
                # loss = (-soft_labels * log_probs).sum(dim=1).mean()
                loss = 0.5 * loss


            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_personalize(args, train_dataloader, model, transfer_parameter_names):

    model_config, fl_config = args[0], args[1]

      

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]

    # optimizer_grouped_parameters = [
    #     {'params':[p for n, p in model.named_parameters() if n not in transfer_parameter_names], 'lr':model_config.learning_rate},
    #     {'params':[p for n, p in model.named_parameters() if n in transfer_parameter_names], 'lr':model_config.transfer_learning_rate}
    # ]

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

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

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _

def train_personalize_cl(args, train_dataloader, model, cur_epoch, trainable_name_list):

    model_config, fl_config = args[0], args[1]

      

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad and n in trainable_name_list],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad and n in trainable_name_list], "weight_decay": 0.0},
    ]

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

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

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
    if fl_config.client_cl:
        thredshould = data_peace_func(fl_config, 0.6, model_config.beta, train_dataloader, cur_epoch, fl_config.rounds)
    else:
        thredshould = len(train_dataloader)
    print("the thredshould steps for this round {}/{}:".format(thredshould, len(train_dataloader)))
    for _ in train_iterator:
        
        for step, batch in enumerate(epoch_iterator):
            if step > thredshould:
                break
            fl_config.logging_steps = len(epoch_iterator)
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch[1])
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            inputs["mask_pos"] = batch[-2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                model.zero_grad()
                global_step += 1

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_personalize_with_our_mask(args, train_dataloader, model, cur_epoch, mask):

    model_config, fl_config = args[0], args[1]

      

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

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

    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
    if fl_config.client_cl:
        thredshould = data_peace_func(fl_config, model_config.init_ratio, model_config.beta, train_dataloader, cur_epoch, fl_config.rounds)
    else:
        thredshould = len(train_dataloader)
    print("the thredshould steps for this round {}/{}:".format(thredshould, len(train_dataloader)))
    for _ in train_iterator:
        
        for step, batch in enumerate(epoch_iterator):
            if step > thredshould:
                break
            fl_config.logging_steps = len(epoch_iterator)
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch[1])
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[-1],
            }
            inputs["token_type_ids"] = batch[2]
            inputs["mask_pos"] = batch[-2]
            inputs["lora_mask"] = mask
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                model.zero_grad()
                global_step += 1

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_personalize_with_mask(args, train_dataloader, model, freeze_parameters_list):

    model_config, fl_config = args[0], args[1]

      

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad and n in freeze_parameters_list],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad and n in freeze_parameters_list], "weight_decay": 0.0},
    ]

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

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

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()      

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_personalize_with_gate(args, train_dataloader, model, gate_model):

    model_config, fl_config = args[0], args[1]

      

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]



    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

    optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)
    optimizer_gate = AdamW(gate_model.parameters(), lr=fl_config.learning_rate)


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

            embeddings = model.get_embeddings_only(**inputs)
            embeddings = torch.mean(embeddings, dim=1)
            # print(embeddings.size())
            # exit()
            gate_model = gate_model.cuda()
            gate_index, gate_scores = gate_model(embeddings)

            gate_index = torch.mean(gate_index, dim=0)
            labels = torch.argsort(-gate_index)[:12]
            labels = labels.unsqueeze(0) 
            target = torch.zeros(labels.size(0), 24).cuda().scatter_(1, labels, 1.)
            mask = target.squeeze_()
            # mask = (target * gate_index).squeeze_()
            inputs["lora_mask"] = mask
            # print(test_parameters)
            outputs = model(**inputs)

            
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # 
                loss.backward()
                

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(gate_model.parameters(), fl_config.max_grad_norm)

                # for name, p in gate_model.named_parameters():
                #     print(name, p.grad)
                # exit()
                # print(test_parameters.grad)
                # exit()
                optimizer_gate.step()
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                gate_model.zero_grad()
                global_step += 1
    
    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, _


def train_personalize_with_prune_growth(args, train_dataloader, model, mask, epoch):

    model_config, fl_config = args[0], args[1]

    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]

    
    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total

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

            # if fl_config.gradient_accumulation_steps > 1:
            #     loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.gradient_accumulation_steps > 1:
                loss = loss / fl_config.gradient_accumulation_steps

            if fl_config.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                model.zero_grad()
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)
                
                # apply the gradient mask
                
                
                # exit()
                for name, p in model.named_parameters():
                    if name in mask:
                        p.grad = p.grad * mask[name].cuda()
                optimizer.step()
                global_step += 1

        if epoch % 10 == 0:
            weights = {k: abs(v.grad.clone().cpu().detach().numpy())
               for k, v in model.named_parameters()
               if v.requires_grad}

            mask = {k: np.ones_like(v)
                    for k, v in weights.items()}
            
            # flat the weights
            weight_flat = np.concatenate([v.flatten() for k, v in weights.items()])

            # get the thredsheld
            number_of_weights_to_prune = int(np.ceil(0.2 * weight_flat.shape[0]))
            threshold = np.sort(np.abs(weight_flat))[number_of_weights_to_prune]

            # get the prune mask
            mask = {k: np.where(np.abs(v) > threshold, mask[k], np.zeros_like(v))
                        for k, v in weights.items()}
            mask = {k: torch.Tensor(v) for k, v in mask.items()}

    # state_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         state_dict[name] = copy.deepcopy(p.data)

    return copy.deepcopy(model.get_copy_of_trainable_weights()), tr_loss / global_step, mask


def evaluate_mask(args, train_dataloader, model):

    model_config, fl_config = args[0], args[1]


    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    

    optimizer_grouped_parameters_transfer = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad ],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters_transfer, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    # )
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
        int(fl_config.mask_epochs),
        desc="Epoch",
        disable=fl_config.local_rank not in [-1, 0],
    )

    set_seed(model_config)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name.lower())
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    global_est_fisher_info = {}

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
        est_fisher_info = {}
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch[1])
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

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)

                # accumulate the fisher information matrix

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
        # evaluate the FIM for rest parameters
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch[1])
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
                model.zero_grad()
                loss.backward()      

            torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)

            # accumulate the fisher information matrix
            for n, p in model.named_parameters(): 
                if p.requires_grad: 
                    # n = n.replace('.', '__') 
                    if p.grad is not None:
                        if n not in est_fisher_info:
                            est_fisher_info[n] = 0
                        est_fisher_info[n] += p.grad.detach() ** 2

        est_fisher_info = {n: p/len(epoch_iterator) for n, p in est_fisher_info.items()}
        # update the momentum of FIM
        for k,v in est_fisher_info.items():
            if k not in global_est_fisher_info:
                global_est_fisher_info[k] = copy.deepcopy(v)
            else:
                global_est_fisher_info[k] = (1 - fl_config.momentum) * global_est_fisher_info[k] + fl_config.momentum * v
    trainable_layer_score = [0 for i in range(24)]

    for k, v in global_est_fisher_info.items():
        layer_index = int(k.split(".")[3])
        trainable_layer_score[layer_index] += torch.sum(v)
    
    print(trainable_layer_score)
    trainable_layer_list = list(torch.argsort(torch.Tensor(trainable_layer_score)).numpy())
    print(trainable_layer_list)

    return trainable_layer_list


def evaluate_mask_layer(args, train_dataloader, model, per_layer_index):

    model_config, fl_config = args[0], args[1]


    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    

    optimizer_grouped_parameters_transfer = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad ],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters_transfer, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    # )
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
        int(fl_config.mask_epochs),
        desc="Epoch",
        disable=fl_config.local_rank not in [-1, 0],
    )

    set_seed(model_config)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name.lower())
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    # global_est_fisher_info = {}
    fisher_layer_score = {i:{"value":None, "query":None} for i in per_layer_index}
    for _ in train_iterator:
        # before each iteration, get the copy of trainable parameters
        old_trainable_params = model.get_copy_of_trainable_weights()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
        est_fisher_info = {}
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(model_config.device) for t in batch[1])
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

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)

                # accumulate the fisher information matrix

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        prefix_name = "roberta.encoder.layer."
        value_name = ".attention.self.value."
        query_name = ".attention.self.query."
        new_params_dict = model.get_copy_of_trainable_weights()
        

        for i in per_layer_index:
            value_A = prefix_name + str(i) + value_name + "lora_A"
            value_B = prefix_name + str(i) + value_name + "lora_B"
            query_A = prefix_name + str(i) + query_name + "lora_A"
            query_B = prefix_name + str(i) + query_name + "lora_B"

            matrix_value_old = old_trainable_params[value_A].transpose(0,1) @ old_trainable_params[value_B].transpose(0,1)
            matrix_value_new = new_params_dict[value_A].transpose(0,1) @ new_params_dict[value_B].transpose(0,1)

            matrix_query_old = old_trainable_params[query_A].transpose(0,1) @ old_trainable_params[query_B].transpose(0,1)
            matrix_query_new = new_params_dict[query_A].transpose(0,1) @ new_params_dict[query_B].transpose(0,1)

            grad_diff_value = abs(matrix_value_new - matrix_value_old) ** 2
            grad_diff_value_mean = torch.mean(grad_diff_value, dim=0)
            grad_diff_query = abs(matrix_query_old - matrix_query_new) ** 2
            grad_diff_query_mean = torch.mean(grad_diff_query, dim=0)

            if fisher_layer_score[i]['value'] == None:
                fisher_layer_score[i]['value'] = grad_diff_value_mean
            else:
                fisher_layer_score[i]['value'] = (1 - fl_config.momentum) * fisher_layer_score[i]['query'] + fl_config.momentum * grad_diff_value_mean
            
            if fisher_layer_score[i]['query'] == None:
                fisher_layer_score[i]['query'] = grad_diff_value_mean
            else:
                fisher_layer_score[i]['query'] = (1 - fl_config.momentum) * fisher_layer_score[i]['query'] + fl_config.momentum * grad_diff_query_mean
            # print(fisher_layer_score[i]['value'].size())

    tmp_list = []
    for i in per_layer_index:
        tmp_list.append(fisher_layer_score[i]['value'])
        tmp_list.append(fisher_layer_score[i]['query'])

    # print(per_layer_index)
    
    # flat the weights
    # weight_flat = torch.concatenate(tmp_list)
    # print(tmp_list)
    weight_flat = torch.cat(tmp_list)

    # get the thredsheld
    number_of_weights_to_prune = int(np.ceil(fl_config.prune_ratio * weight_flat.size()[0]))
    weight_flat_sort, _ = torch.sort(weight_flat)
    threshold = weight_flat_sort[number_of_weights_to_prune]
    # print(threshold)
    # exit()

    sum_mask = 0
    for i in per_layer_index:
        fisher_layer_score[i]['value'] = torch.where(fisher_layer_score[i]['value'] > threshold, 1, 0)
        fisher_layer_score[i]['query'] = torch.where(fisher_layer_score[i]['query'] > threshold, 1, 0)

        sum_mask = sum_mask + torch.sum(fisher_layer_score[i]['value']) + torch.sum(fisher_layer_score[i]['query'])

    print("sparsity of new_mask: {}".format(sum_mask / weight_flat.size()[0]))
    # print(fisher_layer_score)
    # exit()

    return fisher_layer_score

def generate_mask(args, train_dataloader, model):

    model_config, fl_config = args[0], args[1]


    t_total = len(train_dataloader) // fl_config.gradient_accumulation_steps * fl_config.num_local_train_epochs


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    

    optimizer_grouped_parameters_transfer = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad ],
            "weight_decay": fl_config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters_transfer, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    # )
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
        int(2),
        desc="Epoch",
        disable=fl_config.local_rank not in [-1, 0],
    )

    set_seed(model_config)  # Added here for reproductibility
    metric_key = get_metric_key(model_config.task_name.lower())
    if model_config.task_name == 'mnli':
        metric_key = 'avg_acc'

    global_est_fisher_info = {}

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=fl_config.local_rank not in [-1, 0])
        est_fisher_info = {}
        for step, batch in enumerate(epoch_iterator):
            fl_config.logging_steps = len(epoch_iterator)
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

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)

                # accumulate the fisher information matrix

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
    weights = {k: v.clone().cpu().detach().numpy()
            for k, v in model.named_parameters()
            if v.requires_grad}
    # print(weights)
    # exit()

    mask = {k: np.ones_like(v)
            for k, v in weights.items()}
    
    # flat the weights
    weight_flat = np.concatenate([v.flatten() for k, v in weights.items()])

    # get the thredsheld
    number_of_weights_to_prune = int(np.ceil(0.1 * weight_flat.shape[0]))
    print(number_of_weights_to_prune)
    threshold = np.sort(np.abs(weight_flat))[number_of_weights_to_prune]
    print(threshold)
    # exit()

    # get the prune mask
    new_mask = {k: np.where(np.abs(v) > threshold, mask[k], np.zeros_like(v))
                for k, v in weights.items()}
    
    new_mask = {k: torch.tensor(v) for k, v in new_mask.items()}
    
    n_elements = sum([torch.sum(v) for v in new_mask.values()])
    print("sparsity of new_mask: {}".format(n_elements / len(weight_flat)))

    return new_mask

def peace_func(fl_config, b, alpha, train_dataloader, cur_epoch, total_epoch):

    N = total_epoch
    if fl_config.server_peace_func == "linear":
        thredshould_ratio = b + (1-b) * cur_epoch / (alpha * N)
        print("the thredshold ratio:", thredshould_ratio, b, alpha, cur_epoch, N)
    elif fl_config.server_peace_func == 'sqrt':
        thredshould_ratio = b + (1-b) * min(math.sqrt(cur_epoch/(alpha*N)), 1)
    elif fl_config.server_peace_func == 'norm':
        thredshould_ratio = 1
    
    thredshould_steps = min(math.ceil(len(train_dataloader) * thredshould_ratio), len(train_dataloader))

    return thredshould_steps


def data_peace_func(fl_config, b, alpha, train_dataloader, cur_epoch, total_epoch):

    N = total_epoch
    if fl_config.data_peace_func == "linear":
        thredshould_ratio = b + (1-b) * cur_epoch / (alpha * N)
        print("the thredshold ratio:", thredshould_ratio, b, alpha, cur_epoch, N)
    elif fl_config.data_peace_func == 'sqrt':
        thredshould_ratio = b + (1-b) * min(math.pow(cur_epoch/(alpha*N),2), 1)
    elif fl_config.data_peace_func == 'norm':
        thredshould_ratio = 1
    elif fl_config.data_peace_func == 'shortfomer':
        if cur_epoch < 20:
            thredshould_ratio = 0.4
        elif cur_epoch < 50:
            thredshould_ratio = 0.6
        elif cur_epoch < 80:
            thredshould_ratio = 0.8
        else:
            thredshould_ratio = 1.0
    
    thredshould_steps = min(math.ceil(len(train_dataloader) * thredshould_ratio), len(train_dataloader))

    return thredshould_steps

def train_simple(args, train_dataloader, model, ser_epoch=None):

    print("length of train_loader:", len(train_dataloader))
    # print(train_dataloader)
    # print("length:", len(train_dataloader[0]))
    # print(train_dataloader[0][0][0])
    # exit()

    model_config, fl_config = args[0], args[1]

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

    
    optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)

    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total


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
        if fl_config.client_cl:
            thredshould = data_peace_func(fl_config, 0.6, model_config.beta, train_dataloader, ser_epoch, fl_config.rounds)
        else:
            thredshould = len(train_dataloader)
        print("the thredshould steps for this round:", thredshould)
        for step, batch in enumerate(epoch_iterator):
            if step >= thredshould:
                break
            fl_config.logging_steps = len(epoch_iterator)
            # args.logging_steps = 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            # print(batch[1])
            batch = tuple(t.to(model_config.device) for t in batch[1])
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

            if (step + 1) % fl_config.gradient_accumulation_steps == 0:
                if fl_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), fl_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), fl_config.max_grad_norm)


                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        
    
    state_dict = {}
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            # print(name)
            state_dict[name] = copy.deepcopy(p.data)
    
    local_steps = fl_config.num_local_train_epochs * len(epoch_iterator)
    
    return state_dict, tr_loss / global_step, local_steps

def train_others(args, train_dataset, model, col_func, ser_epoch):

    print("length of train_loader:", len(train_dataset))

    model_config, fl_config = args[0], args[1]

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=fl_config.train_batch_size, collate_fn=col_func)    
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

    
    optimizer = AdamW(optimizer_grouped_parameters, lr=fl_config.learning_rate, eps=fl_config.adam_epsilon, weight_decay=0)

    if fl_config.warmup_steps > 0:
        num_warmup_steps = fl_config.warmup_steps
    else:
        num_warmup_steps = fl_config.warmup_rate * t_total


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
        
        if fl_config.client_cl:
            thredshould = data_peace_func(fl_config, 0.6, 0.5, train_dataloader, ser_epoch, fl_config.rounds)
        else:
            thredshould = len(train_dataloader)
        print("the thredshould steps for this round:", thredshould)
        for step, batch in enumerate(epoch_iterator):
            if step >= thredshould:
                break
            fl_config.logging_steps = len(epoch_iterator)
            # args.logging_steps = 1
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            # print(batch[1])
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