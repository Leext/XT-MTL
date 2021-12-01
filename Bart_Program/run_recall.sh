# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-02" 
# INSPUR1="srun --gres=gpu:V100:1" 

$INSPUR1 python -m Bart_Program.recall --input_dir "./Bart_Program/processed_data" \
                        --save_dir "./Bart_Program/logs" \
                        --model_name_or_path "./Bart_Program/pretrained_model" \
                        --ckpt "Bart_Program/saves/recall_rule_rel/epoch_3" \
                        --rep_fn "func_rel_seq" \
                        --simi_fn "edit" \
                        --device cuda:0 \
                        --recall_dump "./Bart_Program/recall_dump/rule_rel"

# $INSPUR1 python -m Bart_Program.recall --input_dir "./Bart_Program/processed_data" \
#                         --save_dir "./Bart_Program/logs" \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --ckpt "./Bart_Program/saves/recall_func/epoch_9" \
#                         --device cuda:3 \
#                         --recall_dump "./Bart_Program/recall_dump/func" &

# $INSPUR1 python -m Bart_Program.recall --input_dir "./Bart_Program/processed_data" \
#                         --save_dir "./Bart_Program/logs" \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --ckpt "./Bart_Program/saves/recall_rel/epoch_6" \
#                         --device cuda:2 \
#                         --recall_dump "./Bart_Program/recall_dump/rel" &

# $INSPUR1 -c 48 python -m Bart_Program.recall --input_dir "./dataset" \
#                         --save_dir "./Bart_Program/logs" \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --rule \
#                         --parallel 48 \
#                         --rep_fn "func_rel_seq" \
#                         --simi_fn "edit" \
#                         --ckpt "./Bart_Program/saves/recall_rel/epoch_6" \
#                         --recall_dump "./Bart_Program/recall_dump/rule_func_rel"

# $INSPUR1 -c 48 python -m Bart_Program.recall --input_dir "./dataset" \
#                         --save_dir "./Bart_Program/logs" \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --rule \
#                         --parallel 48 \
#                         --rep_fn "rel_seq" \
#                         --simi_fn "edit" \
#                         --ckpt "./Bart_Program/saves/recall_rel/epoch_6" \
#                         --recall_dump "./Bart_Program/recall_dump/rule_rel"
