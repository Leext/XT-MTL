# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-03" 
# INSPUR1="srun --gres=gpu:V100:1" 

$INSPUR1 python -m Bart_Program.recall --input_dir "./Bart_Program/processed_data" \
                        --save_dir "./Bart_Program/logs" \
                        --model_name_or_path "./Bart_Program/pretrained_model" \
                        --ckpt "./Bart_Program/saves/recall/epoch_13" \
                        --recall_dump "./Bart_Program/recall_dump"
