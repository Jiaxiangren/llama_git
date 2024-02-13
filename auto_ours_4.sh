for prune_ratio in 0.05
do
    for mask_epochs in 3
    do
        for momentum in 0.2
        do
            for lr in 4e-4
            do
                for layer_num in 18
                do
                    CUDA_VISIBLE_DEVICES=3 python Fed_llama_ours.py --select_method ours \
                                                                    --select_layer_num ${layer_num} \
                                                                    --sort_type ours \
                                                                    --path ./flearn/configs/ours/mpqa.json \
                                                                    --prune_ratio ${prune_ratio} \
                                                                    --mask_epochs ${mask_epochs} \
                                                                    --momentum ${momentum} \
                                                                    --lr ${lr} | tee ./res/ours/mpqa/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
                done
            done
        done
    done
done