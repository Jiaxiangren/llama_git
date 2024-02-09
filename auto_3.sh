source ~/miniconda3/etc/profile.d/conda.sh
conda activate lora


# v2
for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
do
    python Fed_v2.py --lr 1e-4 --path ./flearn/configs/V2/${dataset}.json | tee ./res/v2/${dataset}.txt
done

