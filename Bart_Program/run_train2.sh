# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-05" 
# INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-27" 

# rel exp
sample=0.1
kbp_sample=0.1

#### run exp train
# input_dir=Bart_Program/processed_data_prompt_cbr/func_rel_k5_rel_exp
# comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise_rel_exp"
# $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
#                             --comment $comment \
#                             --output_dir "./Bart_Program/saves" \
#                             --save_dir "./Bart_Program/logs" \
#                             --type prompt \
#                             --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
#                             --main_task "q_cbr" \
#                             --sample $sample \
#                             --revise \
#                             --kbp \
#                             --kbp_sample $kbp_sample \
#                             --kbp_period 1 \
#                             --kbp_mode "triple" \
#                             --start_valid_epoch 20 \
#                             --num_train_epochs 200 \
#                             --device cuda \
#                             --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# input_dir=Bart_Program/processed_data_rel_exp
# comment="vanilla_real_smp_${sample}_rel_exp"
# $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
#                             --comment $comment \
#                             --output_dir "./Bart_Program/saves" \
#                             --save_dir "./Bart_Program/logs" \
#                             --type default \
#                             --sample $sample \
#                             --start_valid_epoch 20 \
#                             --num_train_epochs 200 \
#                             --device cuda \
#                             --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &

#### run exp valid
input_dir=Bart_Program/processed_data_prompt_cbr/func_rel_k5_rel_exp
comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise_rel_exp_valid"
$INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --sample $sample \
                            --valid \
                            --filter_rels "{'population', 'duration','publication date'}" \
                            --revise \
                            --kbp \
                            --kbp_sample $kbp_sample \
                            --kbp_period 1 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "Bart_Program/saves/mtl_qcbr_rel_0.1_kbp_0.1_real_smp_0.1_revise_rel_exp/epoch_36" > ./Bart_Program/error_log/${comment}.log 2>&1 &


input_dir=Bart_Program/processed_data_prompt_cbr/func_rel_k5
comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise_rel_exp_valid_index"
$INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --sample $sample \
                            --valid \
                            --filter_rels "{'population', 'duration','publication date'}" \
                            --revise \
                            --kbp \
                            --kbp_sample $kbp_sample \
                            --kbp_period 1 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "Bart_Program/saves/mtl_qcbr_rel_0.1_kbp_0.1_real_smp_0.1_revise_rel_exp/epoch_36" > ./Bart_Program/error_log/${comment}.log 2>&1 &

# input_dir=Bart_Program/processed_data
# comment="vanilla_real_smp_${sample}_rel_exp_predict"
# $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
#                             --comment $comment \
#                             --output_dir "./Bart_Program/saves" \
#                             --save_dir "./Bart_Program/logs" \
#                             --type default \
#                             --valid \
#                             --filter_rels "{'population', 'duration','publication date'}" \
#                             --sample $sample \
#                             --start_valid_epoch 20 \
#                             --num_train_epochs 200 \
#                             --device cuda \
#                             --model_name_or_path "Bart_Program/saves/vanilla_real_smp_0.1_rel_exp/epoch_47" > ./Bart_Program/error_log/${comment}.log 2>&1 &