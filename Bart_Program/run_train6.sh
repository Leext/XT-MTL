# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-01" 
INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-08" 



sample=0.01
kbp_sample=0.01
comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_smp_${sample}_revise"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                        --main_task "q_cbr" \
                        --sample $sample \
                        --revise \
                        --kbp \
                        --kbp_sample $kbp_sample \
                        --kbp_period 1 \
                        --kbp_mode "triple" \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                        
sample=0.1
kbp_sample=0.1
comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_smp_${sample}_revise"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                        --main_task "q_cbr" \
                        --sample $sample \
                        --revise \
                        --kbp \
                        --kbp_sample $kbp_sample \
                        --kbp_period 1 \
                        --kbp_mode "triple" \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                        
                        
sample=0.5
kbp_sample=0.5
comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_smp_${sample}_revise"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                        --main_task "q_cbr" \
                        --sample $sample \
                        --revise \
                        --kbp \
                        --kbp_sample $kbp_sample \
                        --kbp_period 1 \
                        --kbp_mode "triple" \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                        
                        
sample=1.0
kbp_sample=0.5
comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_smp_${sample}_revise"
$INSPUR1 python -m Bart_Program.train --input_dir "Bart_Program/processed_data_prompt_cbr/func_rel_k5" \
                        --comment $comment \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --type prompt \
                        --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                        --main_task "q_cbr" \
                        --sample $sample \
                        --revise \
                        --kbp \
                        --kbp_sample $kbp_sample \
                        --kbp_period 1 \
                        --kbp_mode "triple" \
                        --start_valid_epoch 15 \
                        --num_train_epochs 100 \
                        --device cuda \
                        --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                        
                        
