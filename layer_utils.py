from ast import arg
import collections
import json
from lib2to3.pgen2 import token
import logging
import os
import random
from re import L
import sys
import pickle

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import transformers
from torch.utils.data import DataLoader
import copy

from data_loader import PromptDataset, PromptSortDataset
from scipy import linalg
import math
from sklearn.metrics.pairwise import cosine_similarity
from torch import linalg as LA

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate_layer_scores(args, train_dataset, model):


    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=20, collate_fn=train_dataset.collate_fn)    

    set_seed(args)  # Added here for reproductibility

    for index, batch in enumerate(train_dataloader):
        if index == 0:
            break
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "labels": batch[-1],
    }
    inputs["token_type_ids"] = batch[2]
    inputs["mask_pos"] = batch[-2]

    # get the embeddings of the input
    embeddings = model.get_embeddings_only(**inputs)
    embeddings = torch.tensor(embeddings, requires_grad=True)
    
    # calculate adversarial embeddings
    adv_inputs = copy.deepcopy(inputs)
    adv_inputs.update({'embedding_output': embeddings})
    adv_outputs = model.get_output_for_embeddings(**adv_inputs)

    loss = adv_outputs[0]
    loss.backward()

    # calculate the modification based on gradient
    grad_norm = torch.norm(embeddings.grad, p=2)
    scale = 0.5 / grad_norm
    epsilion = embeddings.grad * scale
    adv_embeddings = embeddings + epsilion

    # calculate the eigen values and eigen vectors for both x and x' on each layer
    adv_inputs.update({'embedding_output': adv_embeddings})
    adv_outputs = model.get_output_for_embeddings(**adv_inputs)
    hidden_states = adv_outputs[2]
    print("length of tuple:", len(hidden_states))
    
    adv_score_dict = collections.defaultdict(dict)
    for index, x in enumerate(hidden_states[1:]):
        x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
        x_cov = np.cov(x.detach().cpu().T)
        eig_value, eig_vector = linalg.eig(x_cov)
        eig_value = abs(eig_value)
        sort_idx = eig_value.argsort()[::-1]   
        eigenValues = eig_value[sort_idx]
        eigenVectors = eig_vector[:,sort_idx]
        adv_score_dict[index]['eig_value'] = eigenValues.astype(np.float32)
        adv_score_dict[index]['eig_vectors'] = eigenVectors.astype(np.float32)
    
    # calculate the eigvalues and eigenvectors for normal input
    outputs = model(**inputs)
    hidden_states = outputs[2]
    print("length of tuple:", len(hidden_states))
    
    nor_score_dict = collections.defaultdict(dict)
    for index, x in enumerate(hidden_states[1:]):
        x = x[torch.arange(x.size()[0]), inputs['mask_pos']]
        x_cov = np.cov(x.detach().cpu().T)
        eig_value, eig_vector = linalg.eig(x_cov)
        eig_value = abs(eig_value)
        sort_idx = eig_value.argsort()[::-1]   
        eigenValues = eig_value[sort_idx]
        eigenVectors = eig_vector[:,sort_idx]
        nor_score_dict[index]['eig_value'] = eigenValues.astype(np.float32)
        nor_score_dict[index]['eig_vectors'] = eigenVectors.astype(np.float32)
        
    print("the dot prodcut:", np.dot(nor_score_dict[0]['eig_vectors'][:,0], nor_score_dict[0]['eig_vectors'][:,1]))

    # calculate the generality score
    generality_list = []
    for index in range(24):
        
        eigvalue_diff = abs(adv_score_dict[index]['eig_value'] - nor_score_dict[index]['eig_value'])
        # direction_sim = 1 - np.cos(adv_score_dict[index]['eig_vectors'], nor_score_dict[index]['eig_vectors']) + 1e-12
        direction_sim = np.array([cosine_similarity(adv_score_dict[index]['eig_vectors'][:,i].reshape(1,-1), nor_score_dict[index]['eig_vectors'][:,i].reshape(1,-1)) for i in range(len(nor_score_dict[index]['eig_vectors']))])
        # direction_sim = np.array(cosine_similarity(adv_score_dict[index]['eig_vectors'].astype(np.float32), nor_score_dict[index]['eig_vectors'].astype(np.float32)))
        direction_sim = 1 - direction_sim + 1e-12
        print(direction_sim.shape)
        weights = nor_score_dict[index]['eig_value'] / sum(nor_score_dict[index]['eig_value'])
        score = np.sum(weights * eigvalue_diff * direction_sim).astype(np.float32)
        generality_list.append(score)
    
    return generality_list

def calculate_hessian_copy(inputs, model, embeddings, ori_grad):
    
    # hessian_diagonal L(w + h*e) - 2 * L(w) + L(w - h * e) / h^2
    with torch.no_grad():
        embeddings.zero_()

    h = 0.001
    inputs_1 = copy.deepcopy(inputs)
    inputs_2 = copy.deepcopy(inputs)

    inputs_1.update({'embedding_output': embeddings + h * torch.ones_like(embeddings)})
    inputs_2.update({'embedding_output': embeddings - h * torch.ones_like(embeddings)})
    
    adv_outputs = model.get_output_for_embeddings(**inputs)
    loss = model.get_output_for_embeddings(**inputs_1)[0] - model.get_output_for_embeddings(**inputs)[0]
    loss.backward()

    grad_1 = copy.deepcopy(embeddings.grad)
    with torch.no_grad():
        embeddings.zero_()

    loss = model.get_output_for_embeddings(**inputs)[0] - model.get_output_for_embeddings(**inputs_2)[0]
    loss.backward()
    
    grad_2 = copy.deepcopy(embeddings.grad)
    with torch.no_grad():
        embeddings.zero_()

    hessian = grad_2 - grad_1 / h

    # print("hessian size:", hessian.size())
    # print("ori graidient size:", ori_grad.size())


    res = (torch.flatten(ori_grad) * torch.flatten(hessian)) @ torch.flatten(ori_grad).T
    return res


def calculate_hessian(inputs, model, embeddings, ori_grad):
    
    # hessian_diagonal L(w + h*e) - 2 * L(w) + L(w - h * e) / h^2
    with torch.no_grad():
        embeddings.zero_()

    h = 0.001
    inputs_1 = copy.deepcopy(inputs)

    inputs_1.update({'embedding_output': embeddings + h * torch.ones_like(embeddings)})
    
    loss = model.get_output_for_embeddings(**inputs)[0]
    loss.backward()

    grad_1 = copy.deepcopy(embeddings.grad)
    with torch.no_grad():
        embeddings.zero_()

    loss = model.get_output_for_embeddings(**inputs_1)[0]
    loss.backward()
    
    grad_2 = copy.deepcopy(embeddings.grad)
    with torch.no_grad():
        embeddings.zero_()

    hessian = (grad_2 - grad_1) / h

    # print("hessian size:", hessian.size())
    # print("ori graidient size:", ori_grad.size())


    res = (torch.flatten(ori_grad) * torch.flatten(hessian)) @ torch.flatten(ori_grad).T
    return res



def evaluate_layer_scores_F_score(args, train_dataset, model, collate_fn):


    # if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir)

    args.train_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=10, collate_fn=collate_fn)    

    set_seed(args)  # Added here for reproductibility

    for index, batch in enumerate(train_dataloader):
        if index == 0:
            break
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "labels": batch[-1],
    }
    inputs["token_type_ids"] = batch[2]
    inputs["mask_pos"] = batch[-2]

    # get the embeddings of the input
    embeddings = model.get_embeddings_only(**inputs)
    embeddings = torch.tensor(embeddings, requires_grad=True)
    # print(embeddings.size())
    # exit()
    
    # calculate adversarial embeddings
    adv_inputs = copy.deepcopy(inputs)
    adv_inputs.update({'inputs_embeds': embeddings})
    adv_outputs = model.get_output_for_embeddings(**adv_inputs)

    loss = adv_outputs[0]
    loss.backward()

    # calculate the modification based on gradient
    grad_norm = torch.norm(embeddings.grad, p=2)
    scale = 0.5 / grad_norm
    # epsilion = embeddings.grad * scale - 1/2 * calculate_hessian(adv_inputs, model, embeddings, copy.deepcopy(embeddings.grad))
    epsilion = embeddings.grad * scale

    # get the adv embeddings
    adv_embeddings = embeddings + epsilion

    # calculate the eigen values and eigen vectors for both x and x' on each layer
    adv_inputs.update({'inputs_embeds': adv_embeddings})
    adv_outputs = model.get_output_for_embeddings(**adv_inputs)
    hidden_states = adv_outputs[3]
    print("length of tuple:", len(hidden_states))
    
    adv_score_dict = collections.defaultdict(dict)
    for index, x in enumerate(hidden_states[1:]):
        x = x[torch.arange(x.size()[0]), inputs['mask_pos']].unsqueeze(0)
        # print("X.shape:", x.size())
        # exit()
        adv_score_dict[index]['matrix_norm'] = LA.matrix_norm(x, dim=(1,2))
    
    outputs = model(**inputs)
    hidden_states = outputs[3]
    print("length of tuple:", len(hidden_states))
    
    nor_score_dict = collections.defaultdict(dict)
    for index, x in enumerate(hidden_states[1:]):
        x = x[torch.arange(x.size()[0]), inputs['mask_pos']].unsqueeze(0)
        nor_score_dict[index]['matrix_norm'] = LA.matrix_norm(x, dim=(1,2))
        
    # print("the dot prodcut:", np.dot(nor_score_dict[0]['eig_vectors'][:,0], nor_score_dict[0]['eig_vectors'][:,1]))

    # calculate the generality score
    generality_list = []
    for index in range(len(hidden_states[1:])):
        
        F_norm_diff = abs(adv_score_dict[index]['matrix_norm'] - nor_score_dict[index]['matrix_norm'])
        relative_diff = F_norm_diff / nor_score_dict[index]['matrix_norm']
        relative_diff = torch.mean(relative_diff)
        generality_list.append(relative_diff.detach().cpu())
    
    return generality_list

    #[ 0  2  1  3 23 16 17 12  5  6  4 11  9 18 15 20 19 10 21  8 22  7 14 13]
    #[ 2  0  1  3  4 23  7  6  5 10  9  8 11 12 13 14 15 16 18 22 21 17 20 19]
    #[ 2  0  1  3  4 23  7  6  5  8  9 10 11 12 13 14 15 18 16 22 17 20 21 19]