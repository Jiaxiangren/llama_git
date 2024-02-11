from http import client
import torch
import numpy as np
import random
from tqdm import tqdm
from flearn.utils.model_utils_ours import evaluate_mask_layer, evaluate_personalized, average_weights, train, train_others, train_personalize_with_our_mask, train_se, train_plgu, train_fedalt
from flearn.utils.process_data_ours import partition_for_score
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
from layer_utils import evaluate_layer_scores_F_score



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

        if self.args.select_method == "random":
            self.layer_index_list = list(range(25))
            random.shuffle(self.layer_index_list)
        elif self.args.select_method == "increase":
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
        if self.args.sort_type in ['ours', 'voc', 'seqreo', 'shortformer', 'loss']:
            self.train_datasets, self.test_loaders = partition_for_score(self.args, self.train_dataset, self.eval_dataset)
        elif self.args.sort_type in ['vanila', 'se', 'plgu', 'fedalt']:
            self.train_loaders, self.test_loaders  = partition(self.args, self.train_dataset, \
                self.eval_dataset)
    
    def calculate_layer_score(self):
        
        layer_final_scores = None
        for index, dataset in enumerate(self.train_datasets):
            layer_scores = evaluate_layer_scores_F_score(self.args, dataset, self.model, self.train_dataset.collate_fn)
            if not layer_final_scores:
                layer_final_scores = layer_scores
            else:
                for index in range(len(layer_scores)):
                    layer_final_scores[index] += layer_scores[index]
            # print(layer_scores)
        layer_sort = np.argsort(np.array(layer_final_scores))
        print(layer_sort)

        self.layer_index_list = layer_sort
        self.per_index_list = self.layer_index_list[self.general_layer_num:]



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
    
    def generate_prompt(self, transfer_layer_index_list):
        self.train_parameters_name = list()
        self.transfer_parameters_name = list()
        self.server_weights = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                layer_index = int(name.split('.')[2])
                if layer_index in transfer_layer_index_list:
                    self.transfer_parameters_name.append(name)
                self.train_parameters_name.append(name)
                param.requires_grad = True
                self.server_weights[name] = copy.deepcopy(param.data)
            else:
                param.requires_grad = False

        all_param = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                all_param += param.numel()
        print('total param is {}'.format(all_param))

    def generate_weights_for_clients(self):
        
        self.client_weights = []
        for _ in range(self.args.num_clients):
            self.client_weights.append(copy.deepcopy(self.model.get_copy_of_trainable_weights()))


    def data_evaluate_and_score_ours(self):
        # returned train_loaders are list: [score, batch]
        self.train_loaders = []
        self.client_score_dict = {}
        file_name = './sorted_dataset/fl/{}/{}_dataset_{}_fisher_{}.pkl'.format(self.args.task_name, \
                        self.args.sort_type, self.args.num_clients, self.args.dirichlet_alpha)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                self.train_loaders = pickle.load(file)
        else:
            for client_index, train_dataset in enumerate(self.train_datasets):
                print("prepossing the dataset for client:{}".format(client_index))
                self.train_loaders.append(evaluate_and_sort_ours_fisher((self.args, self.args), train_dataset, \
                    self.model, client_index, col_func=self.train_dataset.collate_fn))
            with open(file_name, 'wb') as file:
                pickle.dump(self.train_loaders, file)
        
        for i in range(len(self.train_loaders)):
            totoal_score = 0
            for batch in self.train_loaders[i]:
                totoal_score += batch[0]
            self.client_score_dict[i] = (totoal_score / len(self.train_loaders[i]))
        
        print("the length of the training loaders:", len(self.train_loaders))
    
    def data_evaluate_and_score_voc(self):
        # returned train_loaders are list: [score, batch]
        data = []
        self.sorted_train_dataset = []
        self.client_score_dict = {}
        file_name = './sorted_dataset/fl/{}/{}_dataset_{}_{}.pkl'.format(self.args.task_name, \
                        self.args.sort_type, self.args.num_clients, self.args.dirichlet_alpha)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                data = pickle.load(file)
                self.sorted_train_dataset = [ele[0] for ele in data]
                for index, ele in enumerate(data):
                    self.client_score_dict[index] = ele[1]
        else:
            dir_name = './sorted_dataset/fl/{}/'.format(self.args.task_name)
            os.makedirs(dir_name, exist_ok=True)
            # define the global freq
            self.global_dict = {}
            total = 0
            for client_index, train_dataset in enumerate(self.train_datasets):
                self.global_dict, total = evaluate_and_sort_voc_freq(self.global_dict, total, train_dataset)
            for client_index, train_dataset in enumerate(self.train_datasets):
                print("prepossing the dataset for client:{}".format(client_index))
                sorted_dataset, client_score = evaluate_and_sort_voc(self.args, self.args, train_dataset, \
                    self.model, self.tokenizer, self.global_dict, total)
                data.append((sorted_dataset, client_score))
                self.sorted_train_dataset.append(sorted_dataset)
                self.client_score_dict[client_index] = client_score
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)
        
        print("the length of the training loaders:", len(self.sorted_train_dataset))
    
    def data_evaluate_and_score_seqreo(self):
        # returned train_loaders are list: [score, batch]
        data = []
        self.sorted_train_dataset = []
        self.client_score_dict = {}
        file_name = './sorted_dataset/fl/{}/{}_dataset_{}_{}.pkl'.format(self.args.task_name, \
                        self.args.sort_type, self.args.num_clients, self.args.dirichlet_alpha)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                data = pickle.load(file)
                self.sorted_train_dataset = [ele[0] for ele in data]
                for index, ele in enumerate(data):
                    self.client_score_dict[index] = ele[1]
        else:
            for client_index, train_dataset in enumerate(self.train_datasets):
                print("prepossing the dataset for client:{}".format(client_index))
                sorted_dataset, client_score = evaluate_and_sort_seqreo(self.args, self.args, train_dataset, \
                    self.model, self.tokenizer)
                data.append((sorted_dataset, client_score))
                # print(client_index)
                self.sorted_train_dataset.append(sorted_dataset)
                self.client_score_dict[client_index] = client_score
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)
        
        print("the length of the training loaders:", len(self.sorted_train_dataset))
    
    def data_evaluate_and_score_loss(self):
        # returned train_loaders are list: [score, batch]
        data = []
        self.sorted_train_dataset = []
        self.client_score_dict = {}
        file_name = './sorted_dataset/fl/{}/{}_dataset_{}_loss_{}.pkl'.format(self.args.task_name, \
                        self.args.sort_type, self.args.num_clients, self.args.dirichlet_alpha)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                data = pickle.load(file)
                self.sorted_train_dataset = [ele[0] for ele in data]
                for index, ele in enumerate(data):
                    self.client_score_dict[index] = ele[1]
        else:
            # define the global freq
            total = 0
            for client_index, train_dataset in enumerate(self.train_datasets):
                print("prepossing the dataset for client:{}".format(client_index))
                sorted_dataset, client_score = evaluate_and_sort_loss(self.args, self.args, train_dataset, \
                    self.model, self.tokenizer)
                data.append((sorted_dataset, client_score))
                self.sorted_train_dataset.append(sorted_dataset)
                self.client_score_dict[client_index] = client_score
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)
        
        print("the length of the training loaders:", len(self.sorted_train_dataset))

    
    def data_evaluate_and_score_oursvoc(self):
        # returned train_loaders are list: [score, batch]
        self.train_loaders = []
        self.client_score_dict = {}
        file_name = './sorted_dataset/fl/{}/{}_dataset_{}.pkl'.format(self.args.task_name, \
                        self.args.sort_type, self.args.num_clients)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                self.train_loaders = pickle.load(file)
        else:
            for client_index, train_dataset in enumerate(self.train_datasets):
                print("prepossing the dataset for client:{}".format(client_index))
                self.train_loaders.append(evaluate_and_sort_ours_voc((self.args, self.args), train_dataset, \
                    self.model, self.tokenizer, col_func=self.train_dataset.collate_fn))
            with open(file_name, 'wb') as file:
                pickle.dump(self.train_loaders, file)

        for i in range(len(self.train_loaders)):
            totoal_score = 0
            for batch in self.train_loaders[i]:
                totoal_score += batch[0]
            self.client_score_dict[i] = (totoal_score / len(self.train_loaders[i]))
        
        print("the length of the training loaders:", len(self.train_loaders))
    

    def client_train(self, idxs_users, train_dataloaders, local_weights, time_list, transfer_weights, cur_epoch, transfer_name = None):
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
            self.model.update_trainable_weights_from_dict(copy.deepcopy(self.client_weights[idx]))
            self.model.train()
            self.model.update_transfer_weights_from_dict(copy.deepcopy(transfer_weights))
            start = time.time()
            if self.args.sort_type in ["ours"]:
                w, loss, _ = train_personalize_with_our_mask((self.args, self.args), train_dataloaders[idx], self.model, cur_epoch, self.mask[idx])
            elif self.args.sort_type == "vanila":
                w, loss, _ = train((self.args, self.args), train_dataloaders[idx], self.model, self.train_dataset.collate_fn)
            elif self.args.sort_type == "se":
                w, loss, _ = train_se((self.args, self.args), train_dataloaders[idx], self.model, self.train_dataset.collate_fn)
            elif self.args.sort_type == 'plgu':
                w, loss, _ = train_plgu((self.args, self.args), train_dataloaders[idx], self.model, self.train_dataset.collate_fn)
            elif self.args.sort_type == 'fedalt':
                w, loss, _ = train_fedalt((self.args, self.args), train_dataloaders[idx], self.model, transfer_name)
            else:
                w, loss, _ = train_others((self.args, self.args), train_dataloaders[idx], self.model, self.train_dataset.collate_fn, cur_epoch)
            
            local_weights.append([len(train_dataloaders[idx]), copy.deepcopy(w)])
            delta_time = time.time() - start
            time_list.append(delta_time)
            total_loss += loss * len(train_dataloaders[idx])
            print("{}:{:.4f}".format(idx, loss), end=" ")
            self.client_weights[idx] = copy.deepcopy(self.model.get_copy_of_trainable_weights())
        return total_loss / num_current
    
   

    def train(self):

        # 加载模型
        self.load_model()


        # load dataset and user groups
        self.init_data()

        # calculate the layer scores for determining transfer layers
        if self.args.sort_type == "ours":
            self.calculate_layer_score()


        # evaluate difficulty for each sample batch for each clients
        # self.model = self.model.to(self.args.device)
        if self.args.sort_type == "ours":
            self.data_evaluate_and_score_ours()
        elif self.args.sort_type == "voc":
            self.data_evaluate_and_score_voc()
            _, self.test_loaders, _ = partition(self.args, self.train_dataset, \
                    self.eval_dataset)
        elif self.args.sort_type == "seqreo":
            self.data_evaluate_and_score_seqreo()
            _, self.test_loaders, _ = partition(self.args, self.train_dataset, \
                    self.eval_dataset)
        elif self.args.sort_type == "oursvoc":
            self.data_evaluate_and_score_oursvoc()
        elif self.args.sort_type == "shortformer":
            self.data_evaluate_and_score_seqreo()
            _, self.test_loaders, _ = partition(self.args, self.train_dataset, \
                    self.eval_dataset)
        elif self.args.sort_type == 'loss':
            self.data_evaluate_and_score_loss()
            _, self.test_loaders, _ = partition(self.args, self.train_dataset, \
                    self.eval_dataset)

        # generate the prompt parameters
        transfer_layer_index = self.layer_index_list[:self.general_layer_num]
        num_of_trainable_params = self.generate_prompt(transfer_layer_index)
        print("the transfer layer name:", self.transfer_parameters_name)

        # generate weights for clients
        self.generate_weights_for_clients()


        # Training
        train_losses = []
        test_accs = []
        max_times = []
        best_acc = 0
        training_losses = []
        params_list = []
        training_parameters = []


        global_weights = self.model.get_copy_of_trainable_weights()
        self.mask = []

        # figure out the trainable parameters
        if self.args.sort_type == "ours":

            for client_index in range(self.args.num_clients):
                self.model.train()
                self.model.update_trainable_weights_from_dict(copy.deepcopy(global_weights))

                # evaluate the neuron index of each layer
                # layer_dimension = evaluate_mask_layer()
                layer_mask = evaluate_mask_layer((self.args, self.args), self.train_loaders[client_index], self.model, self.per_index_list)
                self.mask.append(layer_mask)
            

        self.model.train()
        self.model.update_trainable_weights_from_dict(copy.deepcopy(global_weights))
        self.reset_seed()
        transfer_weights = self.model.get_copy_of_transfer_weights(self.transfer_parameters_name)
        test_loss, test_acc = evaluate_personalized(self.args, self.model, self.test_loaders, self.tokenizer, self.client_weights, \
            self.transfer_parameters_name)
        
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
                
            if self.args.sort_type in ["ours"]:
                training_loss = self.client_train(idxs_users, self.train_loaders, \
                local_weights, time_list, transfer_weights, epoch)
            elif self.args.sort_type in ["vanila", "se", "plgu"]:
                training_loss = self.client_train(idxs_users, self.train_loaders, \
                local_weights, time_list, transfer_weights, epoch)
            elif self.args.sort_type in ['fedalt']:
                training_loss = self.client_train(idxs_users, self.train_loaders, \
                local_weights, time_list, transfer_weights, epoch, self.transfer_parameters_name)
            else:
                training_loss = self.client_train(idxs_users, self.sorted_train_dataset, \
                local_weights, time_list, transfer_weights, epoch)


            global_weights = average_weights(local_weights)
            self.model.train()
            self.model.update_trainable_weights_from_dict(copy.deepcopy(global_weights))
            # print(global_weights)

            test_loss, test_acc = evaluate_personalized(self.args, self.model, self.test_loaders, self.tokenizer, self.client_weights, \
                self.transfer_parameters_name)
            
            
            self.model.train()
            self.model.update_trainable_weights_from_dict(copy.deepcopy(global_weights))
            transfer_weights = self.model.get_copy_of_transfer_weights(self.transfer_parameters_name)
                # test_acc = best_acc

            
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
