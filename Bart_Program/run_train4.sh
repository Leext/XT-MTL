# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-01" 
INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-14,dell-gpu-27" 

sample=0.01

comment="vanilla_pretrain_smp_$sample"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data/" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type default \
                        --sample $sample \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "Bart_Program/saves/kb_pretrain/epoch_2" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_qcbr_rel_func_kld_0.1_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1,q_func:func:0.1" \
#                         --kld \
#                         --kld_weight 1 \
#                         --main_task "q_cbr" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda:2 \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# comment="mtl_qcbr_rel_func_0.5_smp_$sample"
# $INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
#                         --comment $comment \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --type prompt \
#                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.5,q_func:func:0.5" \
#                         --main_task "q_cbr" \
#                         --sample $sample \
#                         --start_valid_epoch 15 \
#                         --num_train_epochs 100 \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
