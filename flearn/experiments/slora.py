from difflib import diff_bytes
from http import client
from inspect import Parameter
import torch
import numpy as np
import random
from tqdm import tqdm
from flearn.utils.model_utils_ada import evaluate, average_weights, train, pre_train
from flearn.utils.process_data import PromptDataset, partition
from data.process import tasks_num_labels
from transformers import LlamaConfig, LlamaTokenizer
from models.modeling_llama_lora import LlamaForSequenceClassification
from ..utils.fl_score_functions import *
import copy
import os
import time
import math
import pickle



evaluate_metric = {"rte":"acc",
                    "sst-2":"acc",
                    "cola":"mcc",
                    'mrpc':'acc_and_f1',
                    'mpqa':'acc',
                    'qnli':'acc',
                    "subj":"acc",
                    'trec':"acc",
                    "wnli":"acc",
                    "boolq":"acc",
                    "mr": "acc"}


class CentralTraining(object):
    """
    对于聚合后的模型，进行中心化的训练，share_percent 是共享数据集的大小
    """

    def __init__(self, args, share_percent=0, iid=True, unequal=False, result_dir="central"):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.args = args
        self.general_layer_num = self.args.select_layer_num


        self.v = {}

        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args.data_dir = self.args.data_dir + self.args.task_name + '/'
        self.args.output_dir = self.args.output_dir + 'FL/' + self.args.task_name + '/'

        self.layer_index_list = list(range(25))

        # 设置随机种子
        self.reset_seed()



    def reset_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def init_data(self):
        
        self.args.local_rank = self.args.local_rank
        self.train_dataset = PromptDataset(self.args, self.args.task_name.lower(), self.tokenizer, data_type="train")
        self.eval_dataset = PromptDataset(self.args, self.args.task_name.lower(), self.tokenizer, data_type='dev')
        self.train_loaders = partition(self.args, self.train_dataset, \
            self.eval_dataset)


    def load_model(self):

        if "llama" in self.args.model_name_or_path:
            config = LlamaConfig.from_pretrained(        
                        self.args.model_name_or_path,
                        num_labels=tasks_num_labels[self.args.task_name],
                        finetuning_task=self.args.task_name,
                        cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                        output_hidden_states=True,
                        output_attentions=True
                    )
                
            self.tokenizer = LlamaTokenizer.from_pretrained(
                        self.args.model_name_or_path,
                        do_lower_case=self.args.do_lower_case,
                        cache_dir=self.args.cache_dir if self.args.cache_dir else None,        
                    )
            
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(self.tokenizer.pad_token_id)

            self.tokenizer.add_special_tokens({'mask_token': '<mask>'})
            print(self.tokenizer.mask_token_id)
            # exit()
            
            config.apply_lora=self.args.apply_lora
            print(config.apply_lora)

            # config.apply_lora=False
            config.lora_alpha=self.args.lora_alpha
            config.lora_r=self.args.lora_r
            config.lora_dropout=self.args.lora_dropout
            config.apply_adapter = self.args.apply_adapter
            config.adapter_path = self.args.adapter_path
            config.adapter_type = self.args.adapter_type
            config.adapter_size = self.args.adapter_size
            config.apply_bitfit = self.args.apply_bitfit

            self.model = LlamaForSequenceClassification.from_pretrained(
                        self.args.model_name_or_path,
                        config=config,
                        torch_dtype=torch.float16,
                    ).to(self.args.device)
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate_prompt(self):
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        all_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                all_param += param.numel()
        print('total param is {}'.format(all_param))


    def generate_fine_tune(self):
        for name, param in self.model.named_parameters():
            if 'self_attn.q_proj.weight' in name or "self_attn.v_proj.weight" in name:
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False

        all_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                all_param += param.numel()
        print('total param is {}'.format(all_param))
    

    def client_train(self, idxs_users, train_dataloaders, local_weights, time_list, global_lora):
        """
        进行客户端训练
        :param local_v:
        :param local_P:
        :param idxs_users:
        :param global_model:
        :param user_groups:
        :param epoch:
        :param train_dataset:
        :param train_losses:
        :param local_weights:
        :param local_losses:
        :return:
        """
        num_current = 0
        print(len(train_dataloaders))
        for idx in idxs_users:
            num_current += len(train_dataloaders[idx])
        total_loss = 0

        for idx in idxs_users:

            self.model.train()
            self.model.update_trainable_weights_from_dict(copy.deepcopy(global_lora))

            start = time.time()

            w, loss, _ = train((self.args, self.args), train_dataloaders[idx], self.model, self.train_dataset.collate_fn)
            local_weights.append([len(train_dataloaders[idx]), copy.deepcopy(w)])
            delta_time = time.time() - start
            time_list.append(delta_time)
            total_loss += loss * len(train_dataloaders[idx])
            print("{}:{:.4f}".format(idx, loss), end=" ")
        return total_loss / num_current


    def client_pre_train(self, idxs_users, train_dataloaders, local_weights, global_weights):
        """
        进行客户端训练
        :param local_v:
        :param local_P:
        :param idxs_users:
        :param global_model:
        :param user_groups:
        :param epoch:
        :param train_dataset:
        :param train_losses:
        :param local_weights:
        :param local_losses:
        :return:
        """
        num_current = 0
        print(len(train_dataloaders))
        for idx in idxs_users:
            num_current += len(train_dataloaders[idx])
        total_loss = 0

        for idx in idxs_users:

            self.model.train()
            self.model.update_trainable_weights_from_dict(global_weights)

            w, loss, _ = pre_train((self.args, self.args), train_dataloaders[idx], self.model, self.train_dataset.collate_fn)
            local_weights.append([len(train_dataloaders[idx]), copy.deepcopy(w)])
            total_loss += loss * len(train_dataloaders[idx])
            print("{}:{:.4f}".format(idx, loss), end=" ")
        return total_loss / num_current
    
   

    def train(self):

        # 加载模型
        self.load_model()


        # load dataset and user groups
        self.init_data()


        # evaluate difficulty for each sample for each clients
        self.model = self.model.to(self.args.device)

        ############# stage_1, fine-tune the full parameters #############
        num_of_trainable_params = self.generate_fine_tune()

        initial_weights = self.model.get_copy_of_trainable_weights()
        global_weights = copy.deepcopy(initial_weights)

        for epoch in range(5):
            start = time.time()
            local_weights, time_list = [], []
            print(f'\n | Global Training Round : {epoch} |\n')


            # 选择设备，并进行训练
            idxs_users = np.random.choice(range(self.args.num_clients), self.args.m, replace=False)
            # idxs_users = np.random.choice(range(10), self.args.m, replace=False)
                
            training_loss = self.client_pre_train(idxs_users, self.train_loaders, local_weights, global_weights)


            global_weights = average_weights(local_weights)

            self.model.train()
            self.model.update_trainable_weights_from_dict(copy.deepcopy(global_weights))

            test_loss, test_acc = evaluate((self.args, self.args, self.args), self.model, self.tokenizer)
            
            


            print("pre_training: epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f}".format(epoch, test_loss, test_acc, \
                self.args.learning_rate))
        
        # calculate the diff
        diff_weights = {}
        for key in global_weights:
            diff_weights[key] = global_weights[key] - initial_weights[key]
        
        # decompose the weights
        global_lora = {}
        for layer_name, params in diff_weights.items():
            U, S, V = torch.svd(copy.deepcopy(params).to(torch.float32))
            k = 8
            U_k = U[:, :k]
            S_k_sqrt = torch.diag(torch.sqrt(S[:k]))
            V_k = V[:, :k].T  # Note that torch.svd returns V, not V^T

            PartA = (U_k @ S_k_sqrt).transpose(0,1)
            PartB = (S_k_sqrt @ V_k).transpose(0,1)

            # print(PartA.size())
            # print(PartB.size())
            # exit()
            
            layer_name_A = layer_name.replace('weight', 'lora_A')
            layer_name_B = layer_name.replace('weight', 'lora_B')
            global_lora[layer_name_A] = copy.deepcopy(PartA).to(torch.float16)
            global_lora[layer_name_B] = copy.deepcopy(PartB).to(torch.float16)

        self.generate_prompt()


        ############# stage_2, train lora ###############

        # Training
        train_losses = []
        test_accs = []
        max_times = []
        best_acc = 0
        training_losses = []
        params_list = []
        training_parameters = []

            

        self.reset_seed()
        test_loss, test_acc = evaluate((self.args, self.args, self.args), self.model, self.tokenizer)
        
        # print(test_acc.keys())
        print("-train loss:{:.4f} -test acc:{}".format(test_loss, test_acc))
        lr = self.args.learning_rate
        for epoch in range(self.args.rounds):
            start = time.time()
            local_weights, local_losses, local_v, local_P, time_list = [], [], [], [], []
            print(f'\n | Global Training Round : {epoch} |\n')


            # 选择设备，并进行训练
            idxs_users = np.random.choice(range(self.args.num_clients), self.args.m, replace=False)
            # idxs_users = np.random.choice(range(10), self.args.m, replace=False)
                
            training_loss = self.client_train(idxs_users, self.train_loaders, \
            local_weights, time_list, global_lora)


            global_lora = average_weights(local_weights)
            self.model.train()
            self.model.update_trainable_weights_from_dict(copy.deepcopy(global_lora))
            # print(global_weights)

            test_loss, test_acc = evaluate((self.args, self.args, self.args), self.model, self.tokenizer)
            
            test_accs.append(test_acc)
            max_times.append(max(time_list))
            train_losses.append(test_loss)
            training_losses.append(training_loss)
            training_parameters.append(self.general_layer_num)
            # params_list.append(num_of_trainable_params)
            if test_acc > best_acc:
                best_acc = test_acc
            
            


            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}, {},{}".format(epoch, test_loss, test_acc, \
                self.args.learning_rate, time.time() - start, sum(time_list), training_loss))

        save_path = self.args.output_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        res_dict = {"acc":test_accs, "eval_loss": train_losses, "best_acc": best_acc, "training_time":max_times, "training_loss":training_losses, "num_transfer_params":params_list}
        print(res_dict)
        # with open(save_path + '/metrics_{}'.format(self.prompt_config.prompt_type), 'wb') as f:
        #     pickle.dump(res_dict,f)


# if __name__ == "__main__":
#     t = CentralTraining(args, share_percent=10, iid=0, unequal=False, prune_interval=30, prune_rate=0.6, auto_rate=True, auto_mu=False, server_mu=0, client_mu=0)
#     t.train()
