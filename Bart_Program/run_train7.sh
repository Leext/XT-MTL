# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-01" 
INSPUR1="srun --gres=gpu:V100:1" 

sample=0.5
# comment="cbr_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q_cbr:program:1" \
#                         --main_task "q_cbr" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_qcbr_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1,q_cbr:program:1" \
#                         --main_task "q_cbr" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_qcbr_rel_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
#                         --main_task "q_cbr" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_qcbr_rel_func_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1,q_func:func:0.1" \
#                         --main_task "q_cbr" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &


# comment="mtl_rel_func_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1,q_rel:rel:0.1,q_func:func:0.1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# sample=0.5
# comment="mtl_kbp_path_2_0.5_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.5 \
#                         --kbp_period 2 \
#                         --kbp_mode "path" \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_kbp_path_2_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.1 \
#                         --kbp_period 2 \
#                         --kbp_mode "path" \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# sample=0.1
# comment="mtl_kbp_path_2_0.5_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.5 \
#                         --kbp_period 2 \
#                         --kbp_mode "path" \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
# comment="mtl_kbp_path_2_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.1 \
#                         --kbp_period 2 \
#                         --kbp_mode "path" \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
# sample=0.01
# comment="mtl_kbp_path_2_0.5_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.5 \
#                         --kbp_period 2 \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_kbp_path_2_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.1 \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &


sample=0.01
comment="mtl_kbp_0.005_smp_$sample"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1" \
                        --main_task "q" \
                        --sample $sample \
                        --kbp \
                        --kbp_sample 0.005 \
                        --kbp_period 1 \
                        --kbp_mode "triple" \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

comment="mtl_kbp_path_0.005_smp_$sample"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1" \
                        --main_task "q" \
                        --sample $sample \
                        --kbp \
                        --kbp_sample 0.005 \
                        --kbp_period 1 \
                        --kbp_mode "path" \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                          

# sample=1.0
# comment="mtl_kbp_path_2_0.5_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1" \
#                         --main_task "q" \
#                         --sample $sample \
#                         --kbp \
#                         --kbp_sample 0.01 \
#                         --kbp_period 2 \
#                         --kbp_mode "path" \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                      