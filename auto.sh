# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_llama.py --lr 1e-4 --path ./flearn/configs/LORA/webnlg.json



# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_roberta_lora.py --lr 1e-4 --path ./flearn/configs/ours/rte.json


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_lora.py --lr 1e-4 --path ./flearn/configs/ours/rte.json > ./res/log.txt






# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_lora.py --lr 5e-5 --path ./flearn/configs/ours/rte.json

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 5e-5 --path ./flearn/configs/LPT/rte.json

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_v2.py --lr 5e-5 --path ./flearn/configs/V2/rte.json

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_adapter.py --lr 5e-5 --path ./flearn/configs/ADAPTER/rte.json

# for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
# do
#     CUDA_VISIBLE_DEVICES=7 python Fed_model_ours.py --lr 1e-4 --select_method ours --select_layer_num 24 --sort_type fedalt --path ./flearn/configs/ours/${dataset}.json | tee ./res/exp/fedalt/${dataset}.txt
# done

# for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
# do
#     CUDA_VISIBLE_DEVICES=7 python Fed_model_ours.py --lr 1e-4 --select_method ours --select_layer_num 24 --sort_type fedalt --path ./flearn/configs/ours/${dataset}.json | tee ./res/exp/fedalt/${dataset}.txt
# done

# for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
# do
#     CUDA_VISIBLE_DEVICES=7 python Fed_model_ours.py --lr 1e-4 --select_method ours --select_layer_num 24 --sort_type fedalt --path ./flearn/configs/ours/${dataset}.json | tee ./res/exp/fedalt/${dataset}.txt
# done

