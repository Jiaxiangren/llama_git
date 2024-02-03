import numpy as np
from scipy.stats import dirichlet
from torch.utils.data import DataLoader, RandomSampler, random_split


def partition(args, train_dataset):
    # args.data_partition_method = 'dirichlet_quantity'
    train_dataloader_list = [None for _ in range(args.num_clients)]
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

    elif args.data_partition_method == 'dirichlet_quantity':
        # args.dirichlet_alpha 默认为 5.0  
        num_clients = args.num_clients
        total_samples = len(train_dataset)
        dirichlet_samples = dirichlet.rvs([args.dirichlet_alpha]*num_clients, size=1)
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

    elif args.data_partition_method == 'dirichlet_label':
        raise NotImplementedError()
    
    else:
        raise NotImplementedError()
    
    return train_dataloader_list