# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-04" 
INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-12" 


# ablatio
function run_ablation() {
    sample=$1
    kbp_sample=$2
    if [ $sample == "1.0" ]; then
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5"
    else
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"
    fi
    # 完全体
    # comment="mtl_rel_cbr_infill+2+_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1,q_rel:rel:0.1" \
    #                         --main_task "q_cbr" \
    #                         --revise \
    #                         --kbp \
    #                         --kbp_weight 0.1 \
    #                         --kbp_sample $kbp_sample \
    #                         --kbp_period 2 \
    #                         --kbp_mode "triple" \
    #                         --start_valid_epoch 30 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &             
    # 去掉知识补全预测
    comment="mtl_rel_cbr_infill+2+_real_smp_${sample}_revise"
    log_file="./Bart_Program/error_log/${comment}.log"
    log_content=`cat $log_file`
    if [[ ! -f $log_file ]] || [[ $log_content =~ "CUDA out of memory" ]] ; then
        $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --revise \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
    fi
                            
    # 去掉关系序列预测
    comment="mtl_cbr_infill+2+_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    log_file="./Bart_Program/error_log/${comment}.log"
    log_content=`cat $log_file`
    if [[ ! -f $log_file ]] || [[ $log_content =~ "CUDA out of memory" ]] ; then
        $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_weight 0.1 \
                            --kbp_sample $kbp_sample \
                            --kbp_period 2 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
    fi

    # 去掉 infill         
    comment="mtl_rel_cbr_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    log_file="./Bart_Program/error_log/${comment}.log"
    log_content=`cat $log_file`
    if [[ ! -f $log_file ]] || [[ $log_content =~ "CUDA out of memory" ]] ; then
        $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_weight 0.1 \
                            --kbp_sample $kbp_sample \
                            --kbp_period 2 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
    fi         
                            
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

function run_infilling_test(){
    sample=$1
    kbp_sample=$2
    if [ $sample == "1.0" ]; then
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5"
    else
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"
    fi
    # infill
    # comment="mtl_infill+_real_smp_${sample}_revise"
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,mask:program:1,replace:program:1,swap:program:1" \
    #                         --main_task "q" \
    #                         --revise \
    #                         --start_valid_epoch 20 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
    # kbp 0.001
    comment="mtl_cbr_infill+2+_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_weight 0.1 \
                            --kbp_sample $kbp_sample \
                            --kbp_period 2 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
    # comment="mtl_rel_cbr_infill+2+_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1,q_rel:rel:0.1" \
    #                         --main_task "q_cbr" \
    #                         --revise \
    #                         --kbp \
    #                         --kbp_weight 0.1 \
    #                         --kbp_sample $kbp_sample \
    #                         --kbp_period 2 \
    #                         --kbp_mode "triple" \
    #                         --start_valid_epoch 30 \
    #                         --num_train_epochs 200 \
    #                         --device cuda:3 \
    #                         --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
}

function run_cbr(){
    sample=$1
    kbp_sample=$2
    if [ $sample == "1.0" ]; then
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5"
    else
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"
    fi

    comment="cbr_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q_cbr:program:1" \
                            --main_task "q_cbr" \
                            --revise \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
}

function run_boost_test(){
    sample=$1
    kbp_sample=$2
    if [ $sample == "1.0" ]; then
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_boost"
    else
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_boost_$sample"
    fi
     # kbp 0.001
    comment="mtl_boost_cbr_infill+2+_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_weight 0.1 \
                            --kbp_sample $kbp_sample \
                            --kbp_period 2 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda:2 \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
    comment="mtl_boost_rel_cbr_infill+2+_kbp2_${kbp_sample}_w0.1_real_smp_${sample}_revise"
    $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
                            --comment $comment \
                            --output_dir "./Bart_Program/saves" \
                            --save_dir "./Bart_Program/logs" \
                            --type prompt \
                            --tasks "q:program:1,mask:program:1,swap:program:1,q_cbr:program:1,q_rel:rel:0.1" \
                            --main_task "q_cbr" \
                            --revise \
                            --kbp \
                            --kbp_weight 0.1 \
                            --kbp_sample $kbp_sample \
                            --kbp_period 2 \
                            --kbp_mode "triple" \
                            --start_valid_epoch 30 \
                            --num_train_epochs 200 \
                            --device cuda:3 \
                            --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1 &
}


# run_ablation 100 0.001
# run_ablation 200 0.001
# run_ablation 500 0.001
# run_vanilla 100
# run_vanilla 200
# run_vanilla 500
# run_infilling_test 100 0.001
# run_infilling_test 200 0.001
# run_infilling_test 500 0.001
# run_infilling_test 0.01 0.01
# run_infilling_test 0.1 0.1
# run_infilling_test 0.5 0.5
# run_infilling_test 1.0 0.5
# INSPUR1="srun -p inspur -w inspur-gpu-03" 
# run_boost_test 100 0.001
# run_boost_test 200 0.001
# run_boost_test 500 0.001
# sleep 2s
run_ablation 100 0.001
run_ablation 200 0.001
run_ablation 500 0.001
run_ablation 0.01 0.01
run_ablation 0.1 0.1
run_ablation 0.5 0.5
run_ablation 1.0 0.5