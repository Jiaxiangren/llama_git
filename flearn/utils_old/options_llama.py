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
    server_min: float = field(default=0.0, metadata={"help":"server min"})
    server_max: float = field(default=1e5, metadata={"help":"server max"})
    num_clients: int = field(default=10, metadata={"help":"the number of users in FL"})
    m: int = field(default=2, metadata={"help":"the number of clients participate each round"})
    data_partition_method: str = field(default="iid", metadata={"help":"the method to partition the dataset"})
    dirichlet_alpha: float = field(default=0.1, metadata={"help":"the dirichlet alpha value"})
    train_batch_size: int = field(default=32, metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=32, metadata={"help":"eval batch size"})
    valid_batch_size: int = field(default=32, metadata={"help":"valid batch size"})
    test_batch_size: int = field(default=32, metadata={"help":"test batch size"})
    warmup_steps: int = field(default=0)
    warmup_rate: float = field(default=0.0, metadata={"help":"warmup rate"})
    evaluate_during_training: int = field(default=1, metadata={"help":"whether to evaulate during training"})
    max_steps: int = field(default=5000, metadata={"help":"max steps"})
    learning_rate: float = field(default=1e-3, metadata={"help":"the learning rate of training process"})
    fp16: bool = field(default=False)
    fp16_opt_level: str = field(default="01")
    max_grad_norm: float = field(default=1)
    adam_epsilon: float = field(default=1e-8)
    not_save_model: int = field(default=0, metadata={"help":"whether to save the model"})
    num_local_train_epochs: int = field(default=1, metadata={"help":"the local epoch of client"})
    n_gpu: int = field(default=1, metadata={"help":"the number of gpu used for training"})
    rounds: int = field(default=20, metadata={"help":"the total round of federated FL"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help":"Number of updates steps to accumulate before performing a backward/update pass."})
    weight_decay: float = field(default=0, metadata={"help":"the weight decay"})
    logging_steps: int = field(default=100, metadata={"help": "log the info per a interval of epochs"})

    train_data: str = field(default="./data/lm_data/", metadata={"help":"the directory to store the train data"})
    valid_data: str = field(default="./data/lm_data/", metadata={"help":"the directory to store the valid data"})
    personalized: int = field(default=0, metadata={"help": "whether to use the personalized"})

    # PEFT parameters
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

    output_dir: str = field(default="./output/", metadata={"help":"the directory of output dir"})
    log_dir: str = field(default="./log/", metadata={"help":"the directory of log dir"})
    task_name: str = field(default='rte', metadata={"help":"the task name"})
    model_name_or_path: str = field(default="openlm-research/open_llama_3b", metadata={"help": "the model name used for training"})
    # model_name_or_path: str = field(default="gpt2-large", metadata={"help": "the model name used for training"})
    seq_len: int = field(default=256, metadata={"help":"max sequence length"})
    cache_dir: str = field(default=None, metadata={"help":"the directory of cache dir"})
    do_lower_case: int = field(default=None, metadata={"help":"if lower case the task name"})
    overwrite_cache: bool = field(default=True)
    seed: int = field(default=42, metadata={"help":"the initialization seed"})
    output_mode: str = field(default="classification", metadata={"help":"the mode to get the prediction"})
    apply_lora: Optional[bool] = field(default=True, metadata={"help": "Whether to apply LoRA or not."},)
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"},)
    lora_r: int = field(default=8, metadata={"help": "LoRA r"},)
    lora_path: Optional[str] = field(default=None,metadata={"help": "The file path of LoRA parameters."},)
    apply_adapter: Optional[bool] = field(default=False,metadata={"help": "Whether to apply adapter or not."},)
    adapter_path: Optional[str] = field(default=None,metadata={"help": "The file path of adapter parameters."},)
    adapter_type: Optional[str] = field(default='houlsby',metadata={"help": "houlsby or pfeiffer"},)
    adapter_size: Optional[int] = field(default=64,metadata={"help": "8, 16, 32, 64"},)
    apply_bitfit: Optional[bool] = field(default=False, metadata={"help": "Whether to apply bitfit or not."},)
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    local_rank: int = field(default=-1, metadata={"help": "whether to use the distributed learning, set no as default"})
    method_type: str = field(default="generative", metadata={"help": "whether to use the distributed learning, set no as default"})
    template_idx: int = field(default=0, metadata={"help": "whether to use the distributed learning, set no as default"})
    deepspeed: str = field(default='flearn/configs/ds_config.json')
    gradient_checkpointing: int = field(default=0)
    load_from_cache: int = field(default=1)
    use_cache: int = field(default=0)
    obj: str = field(default="clm")
    

