source ~/miniconda3/etc/profile.d/conda.sh
conda activate adapter


# adapter
for dataset in qnli sst-2 cola mrpc rte boolq mpqa subj trec mr
do
    python Fed_adapter.py --lr 1e-4 --path ./flearn/configs/ADAPTER/${dataset}.json | tee ./res/adapter/${dataset}.txt
done
