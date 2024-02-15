

for dataset in rte cola mrpc subj 
do
    
    CUDA_VISIBLE_DEVICES=1 python Fed_sLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/slora/${dataset}.txt &
    CUDA_VISIBLE_DEVICES=2 python Fed_adaLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/adalora/${dataset}.txt &
    CUDA_VISIBLE_DEVICES=3 python Fed_deltaLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/deltalora/${dataset}.txt
    wait
done


#attempt
CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 5e-5 --path ./flearn/configs/ATTEMPT/subj.json | tee ./res/attempt/subj.txt &

CUDA_VISIBLE_DEVICES=1 python Fed_lpt.py --lr 1e-5 --path ./flearn/configs/ATTEMPT/cola.json | tee ./res/attempt/cola.txt &

#v2
CUDA_VISIBLE_DEVICES=2 python Fed_v2.py --lr 4e-6 --path ./flearn/configs/V2/cola.json | tee ./res/v2/cola.txt &

# prompt
CUDA_VISIBLE_DEVICES=3 python Fed_lpt.py --lr 4e-6 --path ./flearn/configs/PROMPT/cola.json | tee ./res/prompt/cola.txt &

wait

CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 5e-5 --path ./flearn/configs/PROMPT/subj.json | tee ./res/prompt/subj.txt &

#lora
CUDA_VISIBLE_DEVICES=1 python Fed_lora.py --lr 4e-6 --path ./flearn/configs/LORA/mrpc.json | tee ./res/lora/mrpc.txt &

CUDA_VISIBLE_DEVICES=0 python Fed_lora.py --lr 1e-5 --path ./flearn/configs/LORA/cola.json | tee ./res/lora/cola.txt &

CUDA_VISIBLE_DEVICES=2 python Fed_lora.py --lr 5e-5 --path ./flearn/configs/LORA/subj.json | tee ./res/lora/subj.txt &

#voc
CUDA_VISIBLE_DEVICES=3 python Fed_llama_ours.py --lr 5e-5 --select_method increase --select_layer_num 32 --sort_type voc --path ./flearn/configs/ours/subj.json | tee ./res/voc/subj.txt &
wait

CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 5e-5 --select_method increase --select_layer_num 32 --sort_type voc --path ./flearn/configs/ours/mrpc.json | tee ./res/voc/mrpc.txt &

