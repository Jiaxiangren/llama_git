# CUDA_VISIBLE_DEVICES=0 python Fed_slora.py --lr 7e-6 --select_method ours --select_layer_num 24 --path ./flearn/configs/ours/rte.json


# CUDA_VISIBLE_DEVICES=0 python Fed_adalora.py --lr 7e-6 --select_method ours --select_layer_num 24 --path ./flearn/configs/ours/rte.json

CUDA_VISIBLE_DEVICES=0 python Fed_deltaLora.py --lr 7e-6 --select_method ours --select_layer_num 24 --path ./flearn/configs/ours/rte.json
