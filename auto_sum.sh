# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lora


#lora
for dataset in rte
do
    python Fed_lora.py --lr 1e-4 --path ./flearn/configs/LORA/${dataset}.json | tee ./res/lora/${dataset}.txt
done

# lpt
for dataset in rte
do
    python Fed_lpt.py --lr 1e-4 --path ./flearn/configs/LPT/${dataset}.json | tee ./res/lpt/${dataset}.txt
done

# attempt
for dataset in rte
    python Fed_lpt.py --lr 1e-4 --path ./flearn/configs/ATTEMPT/${dataset}.json | tee ./res/attempt/${dataset}.txt
done

#prompt
for dataset in rte
do
    python Fed_lpt.py --lr 1e-4 --path ./flearn/configs/PROMPT/${dataset}.json | tee ./res/prompt/${dataset}.txt
done

# v2
for dataset in rte
do
    python Fed_v2.py --lr 1e-4 --path ./flearn/configs/V2/${dataset}.json | tee ./res/v2/${dataset}.txt
done