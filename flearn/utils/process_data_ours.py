import os
import sys
import logging
from typing import List, Optional
sys.path.append('../')

import torch
from torch.utils.data import Dataset

from Attempt.data.process import convert_examples_to_features
from Attempt.data.process import output_modes
from Attempt.data.process import processors
import numpy as np
from scipy.stats import dirichlet
from torch.utils.data import DataLoader, RandomSampler, random_split, Subset, SequentialSampler
import random
import copy


# logger = logging.getLogger(__name__)


def partition(args, train_dataset, test_dataset=None, logger=None):
    # 如果num_sample_list是None，说明是训练集，那就按dirichlet分布去sample
    # if num_sample_list is None:
        # args.data_partition_method = 'dirichlet_quantity'
    train_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    # test_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    test_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    
    n_sample_list = [0 for _ in range(args.num_clients)]

    
    if args.data_partition_method == 'iid':
        # 计算每份数据的大小
        subset_size = len(train_dataset) // args.num_clients
        # 计算剩余的数据数量
        remaining_size = len(train_dataset) - subset_size * args.num_clients
        # 计算每份数据的数量列表
        subset_sizes = [subset_size] * args.num_clients
        # 将剩余的数据平均分配到每份数据中
        for i in range(remaining_size):
            subset_sizes[i] += 1
        # 使用 random_split 函数将数据集分割成 args.num_clients 份
        subsets = random_split(train_dataset, subset_sizes)
        # 遍历子集并创建 DataLoader 对象
        print('number of samples')
        for i, subset in enumerate(subsets):
            train_sampler = RandomSampler(subset)
            train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
            print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')
            n_sample_list[i] = len(train_dataloader_list[i].dataset)

    elif args.data_partition_method == 'dirichlet_quantity':
        
        # args.dirichlet_alpha 默认为 5.0  
        num_clients = args.num_clients
        total_samples = len(train_dataset)
        dirichlet_samples = dirichlet.rvs([args.dirichlet_alpha]*num_clients, size=1)
        # train_loader
        client_samples = np.round(dirichlet_samples * total_samples).astype(int)
        subset_sizes = client_samples.squeeze()
        # 多余或不足的个数从最后一个人手里减去
        diff = sum(subset_sizes) - total_samples
        subset_sizes[-1] -= diff
        assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
        # 使用 random_split 函数将数据集分割
        subsets = random_split(train_dataset, subset_sizes)
        # 遍历子集并创建 DataLoader 对象
        print('number of samples')
        for i, subset in enumerate(subsets):
            train_sampler = RandomSampler(subset)
            train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
            print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')
            n_sample_list[i] = len(train_dataloader_list[i].dataset)

        #######################################################################################
        # test_loader
        #######################################################################################
        total_samples = len(test_dataset)
        client_samples = np.round(dirichlet_samples * total_samples).astype(int)
        subset_sizes = client_samples.squeeze()
        # 多余或不足的个数从最后一个人手里减去
        diff = sum(subset_sizes) - total_samples
        subset_sizes[-1] -= diff
        assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
        # 使用 random_split 函数将数据集分割
        subsets = random_split(test_dataset, subset_sizes)
        # 遍历子集并创建 DataLoader 对象
        print('number of samples')
        for i, subset in enumerate(subsets):
            test_sampler = SequentialSampler(subset)
            # test_dataloader_list[i] = DataLoader(subset, sampler=test_sampler, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn)
            test_dataloader_list.append(DataLoader(subset, sampler=test_sampler, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn))
            print(f'Client {i}: {len(test_dataloader_list[i].dataset)}')
            # n_sample_list[i] = len(test_dataloader_list[i].dataset)


    elif args.data_partition_method == 'dirichlet_label':
        labels = np.array(train_dataset.all_labels)
        unique_values = set(labels)
        label_list = list(unique_values)
        # label_list = [440, 3216]
        if os.path.exists(os.path.join(args.data_dir, 'saved_train_subsets.pkl')) \
            and os.path.exists(os.path.join(args.data_dir, 'saved_test_subsets.pkl')):
            print("read from saved train and test subsets")
            train_subsets_list = torch.load(os.path.join(args.data_dir, 'saved_train_subsets.pkl'))
            test_subsets_list = torch.load(os.path.join(args.data_dir, 'saved_test_subsets.pkl'))

            for client_idx in range(args.num_clients):
                subset = train_subsets_list[client_idx]
                train_sampler = SequentialSampler(subset)
                train_dataloader_list[client_idx] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, drop_last=True)
                
            for client_idx in range(args.num_clients):
                subset = test_subsets_list[client_idx]
                test_sampler = SequentialSampler(subset)
                test_dataloader_list[client_idx] = DataLoader(subset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)


        else:
            print("creating new datasets from scratch")
            # partition 训练集
            labels = np.array(train_dataset.all_labels)
            num_class = len(label_list)
            # 决定每个client手里有多少数据
            num_clients = args.num_clients
            total_samples = len(train_dataset)
            dirichlet_samples = dirichlet.rvs([5]*num_clients, size=1)
            client_samples = np.round(dirichlet_samples * total_samples).astype(int)
            subset_sizes = client_samples.squeeze()
            # 多余或不足的个数从最后一个人手里减去
            diff = sum(subset_sizes) - total_samples
            subset_sizes[-1] -= diff
            assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
            _backup = copy.deepcopy(subset_sizes)
            num_data_per_client = subset_sizes
            cls_priors   = np.random.dirichlet(alpha=[args.dirichlet_alpha]*num_class,size=num_clients)
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(labels == i)[0] for i in label_list]
            cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
            sample_idx_per_client = [[] for _ in range(num_clients)]
            while(np.sum(subset_sizes)!=0):
                curr_clnt = np.random.randint(num_clients)
                # If current node is full resample a client
                # print('Remaining Data: %d' %np.sum(clnt_data_list))
                if num_data_per_client[curr_clnt] <= 0:
                    continue
                num_data_per_client[curr_clnt] -= 1
                curr_prior = prior_cumsum[curr_clnt]
                while True:
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    # Redraw class label if trn_y is out of that class
                    if cls_amount[cls_label] <= 0:
                        continue
                    cls_amount[cls_label] -= 1
                    sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                    break
            subset_list = []
            for client_idx in range(num_clients):
                subset = Subset(train_dataset, sample_idx_per_client[client_idx])
                subset_list.append(subset)
                train_sampler = SequentialSampler(subset)
                train_dataloader_list[client_idx] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn, drop_last=True)
            # torch.save(subset_list, os.path.join(args.data_dir, 'saved_train_subsets.pkl'))
            
            #######################################################################################
            # test_loader
            #######################################################################################
            if test_dataset:
                if args.personalization:
                    labels = np.array(test_dataset.all_labels)
                    num_class = len(label_list)
                    # 决定每个client手里有多少数据
                    num_clients = args.num_clients
                    total_samples = len(test_dataset)
                    # 用之前的dirichlet_samples
                    # dirichlet_samples = dirichlet.rvs([5]*num_clients, size=1)
                    client_samples = np.round(dirichlet_samples * total_samples).astype(int)
                    subset_sizes = client_samples.squeeze()
                    # 多余或不足的个数从最后一个人手里减去
                    diff = sum(subset_sizes) - total_samples
                    subset_sizes[-1] -= diff
                    assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
                    _backup = copy.deepcopy(subset_sizes)
                    num_data_per_client = subset_sizes
                    # 用之前的
                    # cls_priors   = np.random.dirichlet(alpha=[args.dirichlet_alpha]*num_class,size=num_clients)
                    # prior_cumsum = np.cumsum(cls_priors, axis=1)
                    idx_list = [np.where(labels == i)[0] for i in label_list]
                    cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
                    sample_idx_per_client = [[] for _ in range(num_clients)]
                    while(np.sum(subset_sizes)!=0):
                        curr_clnt = np.random.randint(num_clients)
                        # If current node is full resample a client
                        # print('Remaining Data: %d' %np.sum(clnt_data_list))
                        if num_data_per_client[curr_clnt] <= 0:
                            continue
                        num_data_per_client[curr_clnt] -= 1
                        curr_prior = prior_cumsum[curr_clnt]
                        while True:
                            cls_label = np.argmax(np.random.uniform() <= curr_prior)
                            # Redraw class label if trn_y is out of that class
                            if cls_amount[cls_label] <= 0:
                                continue
                            cls_amount[cls_label] -= 1
                            sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                            break
                    # 先存一下完整的test_dataset
                    
                    # torch.save(test_dataset, os.path.join(args.data_dir, 'saved_testset.pkl'))

                    subset_list = []
                    for client_idx in range(num_clients):
                        subset = Subset(test_dataset, sample_idx_per_client[client_idx])
                        subset_list.append(subset)
                        test_sampler = SequentialSampler(subset)
                        test_dataloader_list[client_idx] = DataLoader(subset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)
                    # torch.save(subset_list, os.path.join(args.data_dir, 'saved_test_subsets.pkl'))
    else:
        raise NotImplementedError()
    
    # 检查每个client上的类的分布
    
    print("training loaders:")
    for i, loader in enumerate(train_dataloader_list):
        if len(label_list) == 2:
            labels = []
            for batch in loader:
                labels.extend(batch[4])
            l0 = 0
            l1 = 0
            for ids in labels:
                if ids == label_list[0]:
                    l0 += 1
                elif ids == label_list[1]:
                    l1 += 1
                else:
                    raise ValueError()
            print(f"client {i}: label 0 : {l0} label 1 : {l1}")
        # n_sample_list[i] = len(train_dataloader_list[i].dataset)
    if args.personalization:
        print("test loaders:")
        for i, loader in enumerate(test_dataloader_list):
            if len(label_list) == 2:
                labels = []
                for batch in loader:
                    labels.extend(batch[4])
                l0 = 0
                l1 = 0
                for ids in labels:
                    if ids == label_list[0]:
                        l0 += 1
                    elif ids == label_list[1]:
                        l1 += 1
                    else:
                        raise ValueError()
                print(f"client {i}: label 0 : {l0} label 1 : {l1}")
            # n_sample_list[i] = l0 + l1
    
    return train_dataloader_list, test_dataloader_list, n_sample_list


def partition_for_score(args, train_dataset, test_dataset=None, logger=None):
    # 如果num_sample_list是None，说明是训练集，那就按dirichlet分布去sample
    # if num_sample_list is None:
        # args.data_partition_method = 'dirichlet_quantity'
    train_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    # test_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    test_dataloader_list = [copy.deepcopy(1) for _ in range(args.num_clients)]
    
    n_sample_list = [0 for _ in range(args.num_clients)]

    
    if args.data_partition_method == 'iid':
        # 计算每份数据的大小
        subset_size = len(train_dataset) // args.num_clients
        # 计算剩余的数据数量
        remaining_size = len(train_dataset) - subset_size * args.num_clients
        # 计算每份数据的数量列表
        subset_sizes = [subset_size] * args.num_clients
        # 将剩余的数据平均分配到每份数据中
        for i in range(remaining_size):
            subset_sizes[i] += 1
        # 使用 random_split 函数将数据集分割成 args.num_clients 份
        subsets = random_split(train_dataset, subset_sizes)
        # 遍历子集并创建 DataLoader 对象
        print('number of samples')
        for i, subset in enumerate(subsets):
            train_sampler = RandomSampler(subset)
            train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
            print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')
            n_sample_list[i] = len(train_dataloader_list[i].dataset)

    elif args.data_partition_method == 'dirichlet_quantity':
        
        # args.dirichlet_alpha 默认为 5.0  
        num_clients = args.num_clients
        total_samples = len(train_dataset)
        dirichlet_samples = dirichlet.rvs([args.dirichlet_alpha]*num_clients, size=1)
        # train_loader
        client_samples = np.round(dirichlet_samples * total_samples).astype(int)
        subset_sizes = client_samples.squeeze()
        # 多余或不足的个数从最后一个人手里减去
        diff = sum(subset_sizes) - total_samples
        subset_sizes[-1] -= diff
        assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
        # 使用 random_split 函数将数据集分割
        subsets = random_split(train_dataset, subset_sizes)
        # 遍历子集并创建 DataLoader 对象
        print('number of samples')
        for i, subset in enumerate(subsets):
            train_sampler = RandomSampler(subset)
            train_dataloader_list[i] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
            print(f'Client {i}: {len(train_dataloader_list[i].dataset)}')
            n_sample_list[i] = len(train_dataloader_list[i].dataset)

        #######################################################################################
        # test_loader
        #######################################################################################
        total_samples = len(test_dataset)
        client_samples = np.round(dirichlet_samples * total_samples).astype(int)
        subset_sizes = client_samples.squeeze()
        # 多余或不足的个数从最后一个人手里减去
        diff = sum(subset_sizes) - total_samples
        subset_sizes[-1] -= diff
        assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
        # 使用 random_split 函数将数据集分割
        subsets = random_split(test_dataset, subset_sizes)
        # 遍历子集并创建 DataLoader 对象
        print('number of samples')
        for i, subset in enumerate(subsets):
            test_sampler = SequentialSampler(subset)
            # test_dataloader_list[i] = DataLoader(subset, sampler=test_sampler, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn)
            test_dataloader_list.append(DataLoader(subset, sampler=test_sampler, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn))
            print(f'Client {i}: {len(test_dataloader_list[i].dataset)}')
            # n_sample_list[i] = len(test_dataloader_list[i].dataset)


    elif args.data_partition_method == 'dirichlet_label':
        labels = np.array(train_dataset.all_labels)
        unique_values = set(labels)
        label_list = list(unique_values)
        # label_list = [440, 3216]
        if os.path.exists(os.path.join(args.data_dir, 'saved_train_subsets.pkl')) \
            and os.path.exists(os.path.join(args.data_dir, 'saved_test_subsets.pkl')):
            print("read from saved train and test subsets")
            train_subsets_list = torch.load(os.path.join(args.data_dir, 'saved_train_subsets.pkl'))
            test_subsets_list = torch.load(os.path.join(args.data_dir, 'saved_test_subsets.pkl'))

            for client_idx in range(args.num_clients):
                subset = train_subsets_list[client_idx]
                train_sampler = RandomSampler(subset)
                train_dataloader_list[client_idx] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
                
            for client_idx in range(args.num_clients):
                subset = test_subsets_list[client_idx]
                test_sampler = SequentialSampler(subset)
                test_dataloader_list[client_idx] = DataLoader(subset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)


        else:
            print("creating new datasets from scratch")
            # partition 训练集
            labels = np.array(train_dataset.all_labels)
            num_class = len(label_list)
            # 决定每个client手里有多少数据
            num_clients = args.num_clients
            total_samples = len(train_dataset)
            dirichlet_samples = dirichlet.rvs([5]*num_clients, size=1)
            client_samples = np.round(dirichlet_samples * total_samples).astype(int)
            subset_sizes = client_samples.squeeze()
            # 多余或不足的个数从最后一个人手里减去
            diff = sum(subset_sizes) - total_samples
            subset_sizes[-1] -= diff
            assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
            _backup = copy.deepcopy(subset_sizes)
            num_data_per_client = subset_sizes
            cls_priors   = np.random.dirichlet(alpha=[args.dirichlet_alpha]*num_class,size=num_clients)
            prior_cumsum = np.cumsum(cls_priors, axis=1)
            idx_list = [np.where(labels == i)[0] for i in label_list]
            cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
            sample_idx_per_client = [[] for _ in range(num_clients)]
            while(np.sum(subset_sizes)!=0):
                curr_clnt = np.random.randint(num_clients)
                # If current node is full resample a client
                # print('Remaining Data: %d' %np.sum(clnt_data_list))
                if num_data_per_client[curr_clnt] <= 0:
                    continue
                num_data_per_client[curr_clnt] -= 1
                curr_prior = prior_cumsum[curr_clnt]
                while True:
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    # Redraw class label if trn_y is out of that class
                    if cls_amount[cls_label] <= 0:
                        continue
                    cls_amount[cls_label] -= 1
                    sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                    break
            train_subsets_list = []
            for client_idx in range(num_clients):
                subset = Subset(train_dataset, sample_idx_per_client[client_idx])
                train_subsets_list.append(subset)
            #     train_sampler = RandomSampler(subset)
            #     train_dataloader_list[client_idx] = DataLoader(subset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn)
            # torch.save(subset_list, os.path.join(args.data_dir, 'saved_train_subsets.pkl'))
            
            #######################################################################################
            # test_loader
            #######################################################################################
            if test_dataset:
                if args.personalization:
                    labels = np.array(test_dataset.all_labels)
                    num_class = len(label_list)
                    # 决定每个client手里有多少数据
                    num_clients = args.num_clients
                    total_samples = len(test_dataset)
                    # 用之前的dirichlet_samples
                    # dirichlet_samples = dirichlet.rvs([5]*num_clients, size=1)
                    client_samples = np.round(dirichlet_samples * total_samples).astype(int)
                    subset_sizes = client_samples.squeeze()
                    # 多余或不足的个数从最后一个人手里减去
                    diff = sum(subset_sizes) - total_samples
                    subset_sizes[-1] -= diff
                    assert min(subset_sizes) > 0, "try a larger dirichlet alpha"
                    _backup = copy.deepcopy(subset_sizes)
                    num_data_per_client = subset_sizes
                    # 用之前的
                    # cls_priors   = np.random.dirichlet(alpha=[args.dirichlet_alpha]*num_class,size=num_clients)
                    # prior_cumsum = np.cumsum(cls_priors, axis=1)
                    idx_list = [np.where(labels == i)[0] for i in label_list]
                    cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
                    sample_idx_per_client = [[] for _ in range(num_clients)]
                    while(np.sum(subset_sizes)!=0):
                        curr_clnt = np.random.randint(num_clients)
                        # If current node is full resample a client
                        # print('Remaining Data: %d' %np.sum(clnt_data_list))
                        if num_data_per_client[curr_clnt] <= 0:
                            continue
                        num_data_per_client[curr_clnt] -= 1
                        curr_prior = prior_cumsum[curr_clnt]
                        while True:
                            cls_label = np.argmax(np.random.uniform() <= curr_prior)
                            # Redraw class label if trn_y is out of that class
                            if cls_amount[cls_label] <= 0:
                                continue
                            cls_amount[cls_label] -= 1
                            sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                            break
                    # 先存一下完整的test_dataset
                    
                    # torch.save(test_dataset, os.path.join(args.data_dir, 'saved_testset.pkl'))

                    subset_list = []
                    for client_idx in range(num_clients):
                        subset = Subset(test_dataset, sample_idx_per_client[client_idx])
                        subset_list.append(subset)
                        test_sampler = SequentialSampler(subset)
                        test_dataloader_list[client_idx] = DataLoader(subset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn)
                    # torch.save(subset_list, os.path.join(args.data_dir, 'saved_test_subsets.pkl'))
    else:
        raise NotImplementedError()
    # # 如果不是None，说明是测试集，那就按照训练集的数量分布来分配test sample
    
    return train_subsets_list, test_dataloader_list


class PromptDataset(Dataset):
    def __init__(self, args, task, tokenizer, data_type="train", sort_type=None):

        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.data_type = data_type

        features = self.convert_to_features()


        self.all_input_ids = [f.input_ids for f in features]
        self.all_attention_mask = [f.attention_mask for f in features]
        self.all_token_type_ids = [f.token_type_ids for f in features]
        self.all_mask_pos = [f.mask_pos for f in features]


        if data_type != 'test':
            self.all_labels = [f.label for f in features]
        else:
            self.all_labels = None

    

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):
        input_ids = self.all_input_ids[index]
        attention_mask = self.all_attention_mask[index]
        token_type_ids = self.all_token_type_ids[index]
        mask_pos = self.all_mask_pos[index]

        if self.all_labels is not None:
            label = self.all_labels[index]
            return (input_ids, attention_mask, token_type_ids, mask_pos, label)
        else:
            return (input_ids, attention_mask, token_type_ids, mask_pos)
        
    def collate_fn(self, batch_data):
        all_length = [len(item[0]) for item in batch_data]
        max_len = max(all_length)

        batch_input_ids, batch_attention_mask = [], []
        batch_token_type_ids, batch_mask_pos, batch_labels = [], [], []
        for i, item in enumerate(batch_data):
            input_ids = item[0]
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_len - all_length[i])
            batch_input_ids.append(input_ids)

            attention_mask = item[1]
            attention_mask = attention_mask + [0] * (max_len - all_length[i])
            batch_attention_mask.append(attention_mask)

            token_type_ids = item[2]
            token_type_ids = token_type_ids + [self.tokenizer.pad_token_type_id] * (max_len - all_length[i])
            batch_token_type_ids.append(token_type_ids)

            mask_pos = item[3]
            batch_mask_pos.append(mask_pos)

            if self.all_labels is not None:
                label = item[-1]
                batch_labels.append(label)
        
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        batch_mask_pos = torch.tensor(batch_mask_pos, dtype=torch.long)
        if len(batch_labels) != 0:
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            return (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_mask_pos, batch_labels)
        else:
            return (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_mask_pos)

    def convert_to_features(self):

        # if self.args.local_rank not in [-1, 0] and self.data_type == "train":
        #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[self.task.lower()]()
        output_mode = output_modes[self.task.lower()]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            self.args.data_dir,
            "cached_{}_{}_{}_{}".format(
                self.data_type,
                list(filter(None, self.args.model_name_or_path.split("/"))).pop(),
                str(self.args.max_seq_length),
                str(self.task),
            ),
        )
        if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
            
            features = torch.load(cached_features_file)
        else:
            print("Creating features from dataset file at %s", self.args.data_dir)

            if self.data_type == "train":
                examples = processor.get_train_examples(self.args.data_dir)
            elif self.data_type == "dev":
                examples = processor.get_dev_examples(self.args.data_dir)
            elif self.data_type == "test":
                examples = processor.get_test_examples(self.args.data_dir)
            else:
                raise NotImplementedError

            label_map = processor.get_label_map()
            features = convert_examples_to_features(
                examples,
                self.tokenizer,
                label_map=label_map,
                max_length=self.args.max_seq_length,
                output_mode=output_mode,
            )

            if self.args.local_rank in [-1, 0]:
                print("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if self.args.local_rank == 0 and not self.data_type == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        return features
        
class PromptSortDataset(PromptDataset):

    def __init__(self, args, task, tokenizer, score_dataset, data_type="train"):

        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.data_type = data_type

        sorted_dataset = sorted(score_dataset, key=lambda x: x[0])

        self.all_input_ids = [f[1][0] for f in sorted_dataset]
        self.all_attention_mask = [f[1][1] for f in sorted_dataset]
        self.all_token_type_ids = [f[1][2] for f in sorted_dataset]
        self.all_mask_pos = [f[1][3] for f in sorted_dataset]

        if data_type != 'test':
            self.all_labels = [f[1][-1] for f in sorted_dataset]
        else:
            self.all_labels = None
