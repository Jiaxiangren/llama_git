

source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberta


#lora
for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
do
    python Fed_lora.py --lr 1e-4 --path ./flearn/configs/LORA/${dataset}.json | tee ./res/lora/${dataset}.txt
done

# lpt
for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
do
    python Fed_lpt.py --lr 1e-4 --path ./flearn/configs/LPT/${dataset}.json | tee ./res/lpt/${dataset}.txt
done