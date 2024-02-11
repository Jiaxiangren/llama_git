#slora
# CUDA_VISIBLE_DEVICES=0 python Fed_slora.py --lr 7e-6 --select_method ours --select_layer_num 24 --path ./flearn/configs/ours/rte.json

#adalora
# CUDA_VISIBLE_DEVICES=0 python Fed_adalora.py --lr 7e-6 --select_method ours --select_layer_num 24 --path ./flearn/configs/ours/rte.json

#deltalora
# CUDA_VISIBLE_DEVICES=0 python Fed_deltaLora.py --lr 7e-6 --select_method ours --select_layer_num 24 --path ./flearn/configs/ours/rte.json

#voc
# CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 25 --sort_type voc --path ./flearn/configs/ours/rte.json

#SLW
# CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 1e-4 --select_method increase --select_layer_num 25 --sort_type seqreo --path ./flearn/configs/ours/rte.json

#SHORTFORMER
# CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 25 --sort_type shortformer --path ./flearn/configs/ours/rte.json

#vanila
CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 25 --sort_type voc --path ./flearn/configs/ours/rte.json