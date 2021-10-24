# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-02" 
# INSPUR1="srun --gres=gpu:V100:1" 

# $INSPUR1 python -m Bart_Program.train_recall --input_dir "./Bart_Program/processed_data" \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --program_rep "rel" \
#                         --device cuda:0 \
#                         --num_train_epochs 10 \
#                         --batch_size 64 \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --comment "recall_rel" &

# $INSPUR1 python -m Bart_Program.train_recall --input_dir "./Bart_Program/processed_data" \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --program_rep "func_rel" \
#                         --device cuda:2 \
#                         --num_train_epochs 10 \
#                         --batch_size 64 \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --comment "recall_func_rel" &

# $INSPUR1 python -m Bart_Program.train_recall --input_dir "./Bart_Program/processed_data" \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --program_rep "func" \
#                         --device cuda:3 \
#                         --num_train_epochs 10 \
#                         --batch_size 64 \
#                         --model_name_or_path "./Bart_Program/pretrained_model" \
#                         --comment "recall_func" &


$INSPUR1 python -m Bart_Program.train_recall --input_dir "./Bart_Program/processed_data" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --pair \
                        --recall_index "Bart_Program/recall_dump/rule_func_rel" \
                        --k 3 \
                        --rep_fn "func_rel_seq" \
                        --simi_fn "edit" \
                        --device cuda:2 \
                        --num_train_epochs 10 \
                        --batch_size 16 \
                        --model_name_or_path "./Bart_Program/pretrained_model" \
                        --comment "recall_rule_func_rel" &

$INSPUR1 python -m Bart_Program.train_recall --input_dir "./Bart_Program/processed_data" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --pair \
                        --recall_index "Bart_Program/recall_dump/rule_func_rel" \
                        --k 3 \
                        --rep_fn "rel_seq" \
                        --simi_fn "edit" \
                        --device cuda:0 \
                        --num_train_epochs 10 \
                        --batch_size 16 \
                        --model_name_or_path "./Bart_Program/pretrained_model" \
                        --comment "recall_rule_rel" &