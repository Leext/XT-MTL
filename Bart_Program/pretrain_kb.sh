# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-03" 
# INSPUR1="srun --gres=gpu:V100:1 -x dell-gpu-14,dell-gpu-27" 

comment="kb_pretrain"
$INSPUR1 python -m Bart_Program.kb_pretrain --input_dir "dataset" \
                                    --comment $comment \
                                    --output_dir "./Bart_Program/saves" \
                                    --save_dir "./Bart_Program/logs" \
                                    --device cuda \
                                    --num_train_epochs 20 \
                                    --model_name_or_path "./Bart_Program/pretrained_model" > ./Bart_Program/error_log/${comment}.log 2>&1