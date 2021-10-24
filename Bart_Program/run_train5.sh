# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-02" 
INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-14,dell-gpu-27" 

sample=0.1
comment="prompt_mtl_qcbr_rel_func_0.1"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_cbr:program:1,q_rel:rel:1,q_func:func:1" \
                        --main_task "q_cbr" \
                        --sample $sample \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="prompt_mtl_cbr_rel_func_0.1"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q_cbr:program:1,q_rel:rel:1,q_func:func:1" \
                        --main_task "q_cbr" \
                        --sample $sample \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="prompt_mtl_rel_func_0.1"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_rel:rel:1,q_func:func:1" \
                        --main_task "q" \
                        --sample $sample \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &


comment="prompt_mtl_qcbr_rel_func_0.01"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_cbr:program:1,q_rel:rel:1,q_func:func:1" \
                        --main_task "q_cbr" \
                        --sample 0.01 \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="prompt_mtl_cbr_rel_func_0.01"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q_cbr:program:1,q_rel:rel:1,q_func:func:1" \
                        --main_task "q_cbr" \
                        --sample 0.01 \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="prompt_mtl_rel_func_0.01"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_rel:rel:1,q_func:func:1" \
                        --main_task "q" \
                        --sample 0.01 \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="vanilla_0.1"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data/" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type default \
                        --sample 0.1 \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="vanilla_0.01"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data/" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type default \
                        --sample 0.01 \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &