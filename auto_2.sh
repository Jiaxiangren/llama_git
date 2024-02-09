
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberta


# attempt
for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
do
    python Fed_lpt.py --lr 1e-4 --path ./flearn/configs/ATTEMPT/${dataset}.json | tee ./res/attempt/${dataset}.txt
done

#prompt
for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
do
    python Fed_lpt.py --lr 1e-4 --path ./flearn/configs/PROMPT/${dataset}.json | tee ./res/prompt/${dataset}.txt
done