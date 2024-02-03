from unittest import result
import torch
import numpy as np
import random
from tqdm import tqdm
from flearn.utils.model_utils_gen import evaluate_llama, average_weights, train

from flearn.utils.process_data import partition
from flearn.utils.data_utils import FT_Dataset
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaTokenizer
import copy
import os
import time
from torch.utils.data import DataLoader, SequentialSampler

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

    def __init__(self, args):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_dtype(torch.float16)

        self.args = args
        self.v = {}

        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 设置随机种子
        self.reset_seed()



    def reset_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def init_data(self):

        train_data = FT_Dataset(
        self.args.train_data, self.args.train_batch_size, self.args.seq_len, 
        joint_lm=self.args.obj=='jlm'
    )
    
        valid_data = FT_Dataset(
            self.args.valid_data, self.args.valid_batch_size, self.args.seq_len,
        )

        self.train_loaders, self.test_loaders, _ = partition(self.args, train_data, valid_data)
        if not self.args.personalized:
            self.test_loader  = DataLoader(valid_data, batch_size=self.args.valid_batch_size, num_workers=0, \
                                        shuffle=False, pin_memory=False, drop_last=False, sampler=SequentialSampler(valid_data))

    def load_model(self):

        from models.modeling_llama_lora import LlamaForCausalLM


        config = LlamaConfig.from_pretrained(        
                    self.args.model_name_or_path,
                    finetuning_task=self.args.task_name,
                    cache_dir=self.args.cache_dir if self.args.cache_dir else None,
                    output_hidden_states=True,
                    output_attentions=True
                )
        config.apply_lora=self.args.apply_lora
        config.lora_alpha=self.args.lora_alpha
        config.lora_r=self.args.lora_r
        config.apply_adapter = self.args.apply_adapter
        config.adapter_path = self.args.adapter_path
        config.adapter_type = self.args.adapter_type
        config.adapter_size = self.args.adapter_size
        config.apply_bitfit = self.args.apply_bitfit

        
        self.model = LlamaForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            local_files_only=False,
            config=config,
        )

        
        self.tokenizer = LlamaTokenizer.from_pretrained(
        self.args.model_name_or_path,
        use_fast=False,
        padding_side='left'
    )
        for p in self.model.parameters():
            p.requires_grad = False
        
        # self.model.generate_prompt_embeddings()
        

        # parameters = model.parameters()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        # self.model, _, _, _ = deepspeed.initialize(
        #     config=self.args.deepspeed,
        #     model=model,
        #     optimizer=optimizer,
        #     model_parameters=parameters
        # )

        # get_accelerator().empty_cache()



    
    def generate_prompt(self):
        # self.model.generate_prompt_embeddings()

        total_model_params = 0
        num_trained_params = 0
        for name, p in self.model.named_parameters():
            if "lora" in name:
                p.requires_grad = True
                num_trained_params += p.numel()
            total_model_params += p.numel()

        print("Total Model Parameters: {}, Trainable Parameters: {}".format(total_model_params, num_trained_params))


    def client_train(self, idxs_users, epoch, train_dataloaders, local_weights, time_list):
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
        for idx in idxs_users:
            num_current += len(train_dataloaders[idx])
        total_loss = 0
        
        # self.fl_config.lr = self.fl_config.init_lr * pow(self.fl_config.lr_decay, epoch)
        ori_trainable_weights = self.model.get_copy_of_trainable_weights()
        for idx in idxs_users:
            start = time.time()
            
            w, loss = train(self.args, self.model, self.tokenizer ,train_dataloaders[idx])
            # if loss < train_losses[0] * 3:
            local_weights.append([len(train_dataloaders[idx]), copy.deepcopy(w)])
            delta_time = time.time() - start
            time_list.append(delta_time)
            total_loss += loss * len(train_dataloaders[idx])
            print("{}:{:.4f}".format(idx, loss), end=" ")
            self.model.update_trainable_weights_from_dict(ori_trainable_weights)
        return total_loss / num_current
        # print("本轮设备总用时：{:.4f}".format(time.time() - start))

    def train(self):
        # 记录日志和结

        # 加载模型
        self.load_model()


        # load dataset and user groups
        self.init_data()
        self.generate_prompt()
        self.model = self.model.cuda()

        # exit()

        


        # exit()
        # Training
        valid_loss_list = []
        valid_ppl_list = []
        max_times = []
        best_val_ppl = 0
        training_losses = []
        params_list = []

        self.reset_seed()

        # 第一次评估
        if not self.args.personalized:
            valid_loss, valid_ppl = evaluate_llama(self.args, self.model, self.test_loader)
        else:
            valid_loss, valid_ppl = evaluate_llama(self.args, self.model, self.test_load)
        # print(test_acc.keys())
        print("-valid loss:{} -valid ppl:{}".format(valid_loss, valid_ppl))
        # print("res:", result)

        # exit()

        for epoch in range(self.args.rounds):
            start = time.time()
            local_weights, time_list = [], []
            print(f'\n | Global Training Round : {epoch} |\n')


            # 选择设备，并进行训练
            self.model.train()
            idxs_users = np.random.choice(range(self.args.num_clients), self.args.m, replace=False)
            training_loss = self.client_train(idxs_users, epoch, self.train_loaders, local_weights, time_list)


            # update global weights
            print("use fedavg as aggregation method on server")
            global_weights = average_weights(local_weights)
            self.model.update_trainable_weights_from_dict(global_weights)
            


            valid_loss, valid_ppl = evaluate_llama(self.args, self.model, self.tokenizer)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            valid_ppl_list.append(valid_ppl)
            valid_loss_list.append(valid_loss)
            max_times.append(max(time_list))
            training_losses.append(training_loss)
            
            


            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}, {},{}".format(epoch, valid_loss, valid_ppl, \
                self.args.learning_rate, time.time() - start, sum(time_list), training_loss))

        save_path = self.args.output_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        res_dict = {"acc":valid_ppl_list, "eval_loss": valid_loss_list, "best_acc": best_val_ppl, "training_time":max_times, "training_loss":training_losses}
        print(res_dict)
        # with open(save_path + '/metrics_{}'.format(self.prompt_config.prompt_type), 'wb') as f:
        #     pickle.dump(res_dict,f)


# if __name__ == "__main__":
#     t = CentralTraining(args, share_percent=10, iid=0, unequal=False, prune_interval=30, prune_rate=0.6, auto_rate=True, auto_mu=False, server_mu=0, client_mu=0)
#     t.train()
