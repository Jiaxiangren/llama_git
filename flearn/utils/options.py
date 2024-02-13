from curses import meta
from dataclasses import dataclass, field
from email.policy import default
from typing import List, Optional
import typing
from xml.etree.ElementInclude import default_loader
from xmlrpc.client import Boolean, boolean


@dataclass
class flArguments():
    share_percent: float = field(default=0.0, metadata={"help":"the shared percentage of the data between clients and server"})
    result_dir: str = field(default="./", metadata={"help":"the directory of result"})
    iid: int = field(default=0, metadata={"help":"IID data if set to 1"})
    unequal: int = field(default=1, metadata={"help":"if the clients has equal num of data"})
    server_min: float = field(default=0.0, metadata={"help":"server min"})
    server_max: float = field(default=1e5, metadata={"help":"server max"})
    num_clients: int = field(default=100, metadata={"help":"the number of users in FL"})
    m: int = field(default=10, metadata={"help":"the number of clients participate each round"})
    data_partition_method: str = field(default="iid", metadata={"help":"the method to partition the dataset"})
    dirichlet_alpha: float = field(default=0.1, metadata={"help":"the dirichlet alpha value"})
    train_batch_size: int = field(default=32, metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=32, metadata={"help":"eval batch size"})
    test_batch_size: int = field(default=32, metadata={"help":"test batch size"})
    warmup_steps: int = field(default=0)
    warmup_rate: float = field(default=0, metadata={"help":"warmup rate"})
    evaluate_during_training: int = field(default=1, metadata={"help":"whether to evaulate during training"})
    max_steps: int = field(default=5000, metadata={"help":"max steps"})
    learning_rate: float = field(default=4e-4, metadata={"help":"the learning rate of training process"})
    fp16: bool = field(default=False)
    fp16_opt_level: str = field(default="01")
    max_grad_norm: float = field(default=1.0)
    adam_epsilon: float = field(default=1e-8)
    not_save_model: int = field(default=0, metadata={"help":"whether to save the model"})
    num_local_train_epochs: int = field(default=1, metadata={"help":"the local epoch of client"})
    n_gpu: int = field(default=1, metadata={"help":"the number of gpu used for training"})
    rounds: int = field(default=20, metadata={"help":"the total round of federated FL"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help":"Number of updates steps to accumulate before performing a backward/update pass."})
    weight_decay: float = field(default=0, metadata={"help":"the weight decay"})
    local_rank: int = field(default=-1, metadata={"help": "whether to use the distributed learning, set no as default"})
    logging_steps: int = field(default=100, metadata={"help": "log the info per a interval of epochs"})
    


    output_dir: str = field(default="./output/", metadata={"help":"the directory of output dir"})
    log_dir: str = field(default="./log/", metadata={"help":"the directory of log dir"})
    task_name: str = field(default='rte', metadata={"help":"the task name"})
    # model_name_or_path: str = field(default="roberta-large", metadata={"help": "the model name used for training"})
    model_name_or_path: str = field(default="openlm-research/open_llama_3b", metadata={"help": "the model name used for training"})
    # model_name_or_path: str = field(default="openlm-research/open_llama_7b", metadata={"help": "the model name used for training"})
    max_seq_length: int = field(default=512, metadata={"help":"max sequence length"})
    cache_dir: str = field(default=None, metadata={"help":"the directory of cache dir"})
    do_lower_case: int = field(default=None, metadata={"help":"if lower case the task name"})
    data_dir: str = field(default="./data/lm_data/", metadata={"help":"the directory to store the data"})
    overwrite_cache: bool = field(default=True)
    seed: int = field(default=42, metadata={"help":"the initialization seed"})
    num_labels: int = field(default=2, metadata={"help":"the size of label set"})
    output_mode: str = field(default="classification", metadata={"help":"the mode to get the prediction"})
    apply_lora: Optional[bool] = field(default=True, metadata={"help": "Whether to apply LoRA or not."},)
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"},)
    lora_r: int = field(default=8, metadata={"help": "LoRA r"},)
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"},)
    lora_path: Optional[str] = field(default=None,metadata={"help": "The file path of LoRA parameters."},)
    apply_adapter: Optional[bool] = field(default=False,metadata={"help": "Whether to apply adapter or not."},)
    adapter_path: Optional[str] = field(default=None,metadata={"help": "The file path of adapter parameters."},)
    adapter_type: Optional[str] = field(default='houlsby',metadata={"help": "houlsby or pfeiffer"},)
    adapter_size: Optional[int] = field(default=64,metadata={"help": "8, 16, 32, 64"},)
    apply_bitfit: Optional[bool] = field(default=False, metadata={"help": "Whether to apply bitfit or not."},)
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})


    prompt_layer_list: int = field(default=None, metadata={"help": "the idx of prompt layers"})
    num_prompt_tokens: int = field(default=128, metadata={"help":"the number of prompt tokens added to the layer"})
    add_prompt_layer: int = field(default=0, metadata={"help":"the index of layer the prompt will be added"})
    proj_down_size: int = field(default=128, metadata={"help":"the projection size"})
    generator_type: str = field(default=None, metadata={"help":"the method to generate the prompt"})
    mode: str = field(default="vanilla_pt", metadata={"help":"whether to use generator or not for prompt embeddings"})
    ft_idx_list: str = field(default=None, metadata={"help":"......"})
    prompt_type: str = field(default="single", metadata={"help":"which prompt method to use"})
    prefix_projection: int = field(default=0, metadata={"help":"use the prefix projection to generate prompt"})
    hidden_dropout_prob: float = field(default=0.1, metadata={"help":"the dropout rate for prompt"})
    prefix_hidden_size: int = field(default=512, metadata={"help":"the hidden size of prompt"})

    alpha: int = field(default=0.2, metadata={"help":"the hidden size of prompt"})
    beta: int = field(default=0.8, metadata={"help":"the hidden size of prompt"})
    sort_type: str = field(default='vanila', metadata={"help": "the type of score function"})
    data_peace_func: str = field(default='linear', metadata={"help": "the peace func for CL"})
    server_peace_func: str = field(default='linear', metadata={"help": "the peace func for server CL"})
    server_cl: int = field(default=1, metadata={"help":"if we use the server CL"})
    client_cl: int = field(default=1, metadata={"help": "whether to use client cl"})
    personalization: int = field(default=0, metadata={"help": "personalization"})
    layer_peace_func: str = field(default="linear", metadata={"help":"whether to use the layer peace function"})
    momentum: float = field(default=0.5, metadata={"help":"whether to use the layer peace function"})
    prune_ratio: float = field(default=0.2, metadata={"help":"whether to use the layer peace function"})
    mask_epochs: int = field(default=2, metadata={"help":"whether to use the layer peace function"})
    trainable_size: int = field(default=20, metadata={"help":"whether to use the layer peace function"})
    init_ratio: float = field(default=0.6, metadata={"help":"whether to use the layer peace function"})
    interval: int = field(default=5, metadata={"help": "the interval of full rank matrix aggregation"})

    server_train: int = field(default=1, metadata={"help": "whether to use server optimizer"})
    server_optimizer: str = field(default="Adam", metadata={"help": "the type of optimizer used on server"})
    server_lr: float = field(default=1.0, metadata={"help":"the server lr "})
    increas_ranks: int = field(default=2)
    epsilion: float = field(default=0.02)


    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )

    bf16: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    

