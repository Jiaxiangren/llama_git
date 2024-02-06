# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_llama.py --lr 1e-4 --path ./flearn/configs/LORA/webnlg.json



# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_roberta_lora.py --lr 1e-4 --path ./flearn/configs/ours/rte.json


# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_lora.py --lr 1e-4 --path ./flearn/configs/ours/rte.json > ./res/log.txt



CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python Fed_lora.py --lr 1e-2 --path ./flearn/configs/ours/rte.json