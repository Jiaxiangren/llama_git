#attempt
CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 5e-5 --path ./flearn/configs/ATTEMPT/subj.json | tee ./res/attempt/subj.txt &


#v2
CUDA_VISIBLE_DEVICES=1 python Fed_v2.py --lr 3e-5 --path ./flearn/configs/V2/mrpc.json | tee ./res/v2/mrpc.txt &

# prompt
CUDA_VISIBLE_DEVICES=2 python Fed_lpt.py --lr 3e-5 --path ./flearn/configs/PROMPT/cola.json | tee ./res/prompt/cola.txt &
CUDA_VISIBLE_DEVICES=3 python Fed_lpt.py --lr 5e-5 --path ./flearn/configs/PROMPT/subj.json | tee ./res/prompt/subj.txt &

wait

#lora
CUDA_VISIBLE_DEVICES=1 python Fed_lora.py --lr 5e-5 --path ./flearn/configs/LORA/mrpc.json | tee ./res/lora/mrpc.txt &

CUDA_VISIBLE_DEVICES=2 python Fed_lora.py --lr 1e-5 --path ./flearn/configs/LORA/cola.json | tee ./res/lora/cola.txt &

CUDA_VISIBLE_DEVICES=3 python Fed_lora.py --lr 5e-5 --path ./flearn/configs/LORA/subj.json | tee ./res/lora/subj.txt &

CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 5e-5 --select_method increase --select_layer_num 32 --sort_type voc --path ./flearn/configs/ours/subj.json | tee ./res/voc/subj.txt &

wait
#voc

for prune_ratio in 0.05
do
    for mask_epochs in 3
    do
        for momentum in 0.5
        do
            for lr in 7e-6 3e-6
            do
                for layer_num in 24 28
                do
                    CUDA_VISIBLE_DEVICES=2 python Fed_llama_ours.py --select_method ours \
                                                                    --select_layer_num ${layer_num} \
                                                                    --sort_type ours \
                                                                    --path ./flearn/configs/ours/subj.json \
                                                                    --prune_ratio ${prune_ratio} \
                                                                    --mask_epochs ${mask_epochs} \
                                                                    --momentum ${momentum} \
                                                                    --lr ${lr} | tee ./res/ours/subj/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
                done
            done
        done
    done
done

for prune_ratio in 0.05
do
    for mask_epochs in 3
    do
        for momentum in 0.5
        do
            for lr in 7e-6 3e-6
            do
                for layer_num in 24 28
                do
                    CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --select_method ours \
                                                                    --select_layer_num ${layer_num} \
                                                                    --sort_type ours \
                                                                    --path ./flearn/configs/ours/mrpc.json \
                                                                    --prune_ratio ${prune_ratio} \
                                                                    --mask_epochs ${mask_epochs} \
                                                                    --momentum ${momentum} \
                                                                    --lr ${lr} | tee ./res/ours/mrpc/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
                done
            done
        done
    done
done


for prune_ratio in 0.05
do
    for mask_epochs in 3
    do
        for momentum in 0.5
        do
            for lr in 7e-6 3e-6
            do
                for layer_num in 24 28
                do
                    CUDA_VISIBLE_DEVICES=1 python Fed_llama_ours.py --select_method ours \
                                                                    --select_layer_num ${layer_num} \
                                                                    --sort_type ours \
                                                                    --path ./flearn/configs/ours/cola.json \
                                                                    --prune_ratio ${prune_ratio} \
                                                                    --mask_epochs ${mask_epochs} \
                                                                    --momentum ${momentum} \
                                                                    --lr ${lr} | tee ./res/ours/cola/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
                done
            done
        done
    done
done




CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 5e-5 --select_method increase --select_layer_num 32 --sort_type voc --path ./flearn/configs/ours/mrpc.json | tee ./res/voc/mrpc.txt &


for dataset in rte cola mrpc subj 
do
    
    CUDA_VISIBLE_DEVICES=1 python Fed_sLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/slora/${dataset}.txt &
    CUDA_VISIBLE_DEVICES=2 python Fed_adaLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/adalora/${dataset}.txt &
    CUDA_VISIBLE_DEVICES=3 python Fed_deltaLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/deltalora/${dataset}.txt
    wait
done

