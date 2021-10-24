# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-01" 
INSPUR1="srun --gres=gpu:V100:1" 

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data" \
                        --comment "vanilla2" \
                        --output_dir "./Bart_Program/saves" \
                        --save_dir "./Bart_Program/logs" \
                        --device cuda \
                        --num_train_epochs 30 \
                        --model_name_or_path "./Bart_Program/pretrained_model"