
# for mrpc
for prune_ratio in 0.05
do
    for mask_epochs in 3
    do
        for momentum in 0.5
        do
            for lr in 1e-5 5e-5
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

# rte
for prune_ratio in 0.05
do
    for mask_epochs in 3
    do
        for momentum in 0.5
        do
            for lr in 1e-5 5e-5
            do
                for layer_num in 24 28
                do
                    CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --select_method ours \
                                                                    --select_layer_num ${layer_num} \
                                                                    --sort_type ours \
                                                                    --path ./flearn/configs/ours/rte.json \
                                                                    --prune_ratio ${prune_ratio} \
                                                                    --mask_epochs ${mask_epochs} \
                                                                    --momentum ${momentum} \
                                                                    --lr ${lr} | tee ./res/ours/rte/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
                done
            done
        done
    done
done

#cola



# for prune_ratio in 0.05 0.1 0.2
# do
#     for mask_epochs in 3 4
#     do
#         for momentum in 0.2 0.5 0.8
#         do
#             for lr in 4e-4 6e-4 8e-4
#             do
#                 for layer_num in 12 15 18
#                 do
#                     CUDA_VISIBLE_DEVICES=5 python Fed_model_ours.py --select_method ours \
#                                                                     --select_layer_num ${layer_num} \
#                                                                     --sort_type ours \
#                                                                     --path ./flearn/configs/ours/sst-2.json \
#                                                                     --prune_ratio ${prune_ratio} \
#                                                                     --mask_epochs ${mask_epochs} \
#                                                                     --momentum ${momentum} \
#                                                                     --lr ${lr} | tee ./res/ours_only/sst-2/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
#                 done
#             done
#         done
#     done
# done

#trec


# subj



# mr
# for prune_ratio in 0.05
# do
#     for mask_epochs in 3
#     do
#         for momentum in 0.5
#         do
#             for lr in 1e-5 5e-5
#             do
#                 for layer_num in 24 28
#                 do
#                     CUDA_VISIBLE_DEVICES=4 python Fed_llama_ours.py --select_method ours \
#                                                                     --select_layer_num ${layer_num} \
#                                                                     --sort_type ours \
#                                                                     --path ./flearn/configs/ours/mr.json \
#                                                                     --prune_ratio ${prune_ratio} \
#                                                                     --mask_epochs ${mask_epochs} \
#                                                                     --momentum ${momentum} \
#                                                                     --lr ${lr} | tee ./res/ours/mr/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
#                 done
#             done
#         done
#     done
# done


# boolq
# for prune_ratio in 0.05 0.1 0.2
# do
#     for mask_epochs in 1 3 4
#     do
#         for momentum in 0.2 0.5 0.8
#         do
#             for lr in 1e-4 2e-4
#             do
#                 for layer_num in 12 15 18
#                 do
#                     CUDA_VISIBLE_DEVICES=5 python Fed_model_ours.py --select_method ours \
#                                                                     --select_layer_num ${layer_num} \
#                                                                     --sort_type ours \
#                                                                     --path ./flearn/configs/ours/boolq.json \
#                                                                     --prune_ratio ${prune_ratio} \
#                                                                     --mask_epochs ${mask_epochs} \
#                                                                     --momentum ${momentum} \
#                                                                     --lr ${lr} | tee ./res/ours_only/boolq/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
#                 done
#             done
#         done
#     done
# done


# qnli
# for prune_ratio in 0.05 0.1 0.2
# do
#     for mask_epochs in 3 4
#     do
#         for momentum in 0.2 0.5 0.8
#         do
#             for lr in 4e-4 6e-4 8e-4
#             do
#                 for layer_num in 12 15 18
#                 do
#                     CUDA_VISIBLE_DEVICES=5 python Fed_model_ours.py --select_method ours \
#                                                                     --select_layer_num ${layer_num} \
#                                                                     --sort_type ours \
#                                                                     --path ./flearn/configs/ours/qnli.json \
#                                                                     --prune_ratio ${prune_ratio} \
#                                                                     --mask_epochs ${mask_epochs} \
#                                                                     --momentum ${momentum} \
#                                                                     --lr ${lr} | tee ./res/ours_only/qnli/${prune_ratio}_${mask_epochs}_${momentum}_${lr}_${layer_num}.txt
#                 done
#             done
#         done
#     done
# done

# mpqa


