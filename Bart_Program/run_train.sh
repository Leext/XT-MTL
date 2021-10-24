# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-02" 
INSPUR1="srun --gres=gpu:V100:1" 

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data/" \
                        --comment "vanilla_0.5" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --sample 0.5 \
                        --num_train_epochs 30 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" &

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data_cbr/rule_func_rel_k5" \
                        --comment "cbr_rule_func_rel_k5_0.5" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --cbr \
                        --sample 0.5 \
                        --num_train_epochs 50 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" &

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data_cbr/rule_func_rel_k5" \
                        --comment "cbr_rule_func_rel_k5_0.2" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --cbr \
                        --sample 0.2 \
                        --num_train_epochs 50 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" &

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data_cbr/rule_rel_k5" \
                        --comment "cbr_rule_rel_k5_0.5" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --cbr \
                        --sample 0.5 \
                        --num_train_epochs 50 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" &

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data_cbr/rule_rel_k5" \
                        --comment "cbr_rule_rel_k5_0.2" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --cbr \
                        --sample 0.2 \
                        --num_train_epochs 50 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model"

# $INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data" \
#                         --comment "vanilla_24" \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --num_train_epochs 25 \
#                         --device cuda:0 \
#                         --model_name_or_path "Bart_Program/saves/vanilla/epoch_24" &

wait