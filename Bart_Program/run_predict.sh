# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-05" 
# INSPUR1="srun --gres=gpu:V100:1" 

idx=1

function run_predict() {
    sample=$1
    kbp_sample=$2
    if [ $sample == "1.0" ]; then
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5"
    else
        input_dir="Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"
    fi
    # 完全体
    comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise"
    epoch=`tail -n 4 "./Bart_Program/error_log/${comment}.log" | grep -E $'epoch ([0-9]+)' -o`
    epoch=${epoch:6}
    check_path="./Bart_Program/saves/$comment/epoch_$epoch"
    comment=${comment}_predict
    echo $check_path
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
    #                         --main_task "q_cbr" \
    #                         --revise \
    #                         --predict \
    #                         --kbp \
    #                         --kbp_sample $kbp_sample \
    #                         --kbp_period 1 \
    #                         --kbp_mode "triple" \
    #                         --start_valid_epoch 20 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path $check_path > ./Bart_Program/error_log/${comment}.log 2>&1 
    predict_file="./Bart_Program/saves/$comment/predict.txt"
    cp $predict_file "./Bart_Program/predicts/${idx}.txt"
    idx=$(($idx+1))
    # 去掉知识补全预测
    comment="mtl_qcbr_rel_0.1_real_smp_${sample}_revise"
    epoch=`tail -n 4 "./Bart_Program/error_log/${comment}.log" | grep -E $'epoch ([0-9]+)' -o`
    epoch=${epoch:6}
    check_path="./Bart_Program/saves/$comment/epoch_$epoch"
    comment=${comment}_predict
    echo $check_path
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
    #                         --main_task "q_cbr" \
    #                         --predict \
    #                         --revise \
    #                         --start_valid_epoch 20 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path $check_path > ./Bart_Program/error_log/${comment}.log 2>&1 
    predict_file="./Bart_Program/saves/$comment/predict.txt"
    cp $predict_file "./Bart_Program/predicts/${idx}.txt"
    idx=$(($idx+1))
                            
    # 去掉关系序列预测
    comment="mtl_qcbr_kbp_${kbp_sample}_real_smp_${sample}_revise"
    epoch=`tail -n 4 "./Bart_Program/error_log/${comment}.log" | grep -E $'epoch ([0-9]+)' -o`
    epoch=${epoch:6}
    check_path="./Bart_Program/saves/$comment/epoch_$epoch"
    comment=${comment}_predict
    echo $check_path
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,q_cbr:program:1" \
    #                         --main_task "q_cbr" \
    #                         --revise \
    #                         --predict \
    #                         --kbp \
    #                         --kbp_sample $kbp_sample \
    #                         --kbp_period 1 \
    #                         --kbp_mode "triple" \
    #                         --start_valid_epoch 20 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path $check_path > ./Bart_Program/error_log/${comment}.log 2>&1 
    predict_file="./Bart_Program/saves/$comment/predict.txt"
    cp $predict_file "./Bart_Program/predicts/${idx}.txt"
    idx=$(($idx+1)) 
    # 去掉查询修正
    comment="mtl_qcbr_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}"
    epoch=`tail -n 4 "./Bart_Program/error_log/${comment}.log" | grep -E $'epoch ([0-9]+)' -o`
    epoch=${epoch:6}
    check_path="./Bart_Program/saves/$comment/epoch_$epoch"
    comment=${comment}_predict
    echo $check_path
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,q_cbr:program:1,q_rel:rel:0.1" \
    #                         --main_task "q_cbr" \
    #                         --predict \
    #                         --kbp \
    #                         --kbp_sample $kbp_sample \
    #                         --kbp_period 1 \
    #                         --kbp_mode "triple" \
    #                         --start_valid_epoch 20 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path $check_path > ./Bart_Program/error_log/${comment}.log 2>&1 
    predict_file="./Bart_Program/saves/$comment/predict.txt"
    cp $predict_file "./Bart_Program/predicts/${idx}.txt"
    idx=$(($idx+1))       
                            
    # 去掉CBR
    comment="mtl_rel_0.1_kbp_${kbp_sample}_real_smp_${sample}_revise"
    epoch=`tail -n 4 "./Bart_Program/error_log/${comment}.log" | grep -E $'epoch ([0-9]+)' -o`
    epoch=${epoch:6}
    check_path="./Bart_Program/saves/$comment/epoch_$epoch"
    comment=${comment}_predict
    echo $check_path
    # $INSPUR1 python -m Bart_Program.train --input_dir $input_dir \
    #                         --comment $comment \
    #                         --output_dir "./Bart_Program/saves" \
    #                         --save_dir "./Bart_Program/logs" \
    #                         --type prompt \
    #                         --tasks "q:program:1,q_rel:rel:0.1" \
    #                         --main_task "q" \
    #                         --revise \
    #                         --predict \
    #                         --kbp \
    #                         --kbp_sample $kbp_sample \
    #                         --kbp_period 1 \
    #                         --kbp_mode "triple" \
    #                         --start_valid_epoch 20 \
    #                         --num_train_epochs 200 \
    #                         --device cuda \
    #                         --model_name_or_path $check_path > ./Bart_Program/error_log/${comment}.log 2>&1 
    predict_file="./Bart_Program/saves/$comment/predict.txt"
    cp $predict_file "./Bart_Program/predicts/${idx}.txt"
    idx=$(($idx+1))        
                            
}
run_predict 0.01 0.01
run_predict 0.1 0.1
run_predict 0.5 0.5
run_predict 1.0 0.5