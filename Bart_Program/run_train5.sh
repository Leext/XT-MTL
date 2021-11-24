# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-02" 
INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-12" 


# ablation
function run_ablation() {
    sample=$1
    kbp_sample=$2
    if [ $sample == "1.0" ]; then
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5"
    else
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"
    fi
    # 完全体
    comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_sample $kbp_sample \
                            --kbp_period 1 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                            
    # 去掉知识补全预测
    comment="mtl_qcbr_rel_0.1_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --revise \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                            
    # 去掉关系序列预测
    comment="mtl_qcbr_kbp_${kbp_sample}_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_sample $kbp_sample \
                            --kbp_period 1 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                            
    # 去掉查询修正
    comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --kbp \
                            --kbp_sample $kbp_sample \
                            --kbp_period 1 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                            
    # 去掉CBR
    comment="mtl_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_rel:rel:0.1" \
                            --main_task "q" \
                            --revise \
                            --kbp \
                            --kbp_sample $kbp_sample \
                            --kbp_period 1 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 20 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
                            
}

function run_vanilla(){
    sample=$1
    input_dir=Bart_Program/processed_data_${sample}
    comment="vanilla_real_smp_${sample}"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                                --comment $comment \
                                --output_dir "./Bart_Program/saves" \
                                --save_dir "./Bart_Program/logs" \
                                --type default \
                                --start_valid_epoch 20 \
                                --num_train_epochs 200 \
                                --device cuda \
                                --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
}
# run_ablation 100 0.001
# run_ablation 200 0.001
# run_ablation 500 0.001
run_vanilla 100
run_vanilla 200
run_vanilla 500