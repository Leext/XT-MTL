# /bin/bash
INSPUR1="srun -p inspur -w inspur-gpu-05" 
# INSPUR1="srun --gres=gpu:V100:1" 

# $INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data" \
#                         --comment "vanilla" \
#                         --output_dir "./Bart_Program/saves" \
#                         --save_dir "./Bart_Program/logs" \
#                         --device cuda \
#                         --model_name_or_path "./Bart_Program/pretrained_model"

# $INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data/" \
#                                     --comment "vanilla_0.2" \
#                                     --output_dir "./Bart_Program/saves" \
#                                     --save_dir "./Bart_Program/logs" \
#                                     --sample 0.2 \
#                                     --num_train_epochs 30 \
#                                     --device cuda \
#                                     --model_name_or_path "./Bart_Program/pretrained_model" &

# $INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data/" \
#                                     --comment "vanilla_0.1" \
#                                     --output_dir "./Bart_Program/saves" \
#                                     --save_dir "./Bart_Program/logs" \
#                                     --sample 0.1 \
#                                     --num_train_epochs 30 \
#                                     --device cuda \
#                                     --model_name_or_path "./Bart_Program/pretrained_model" &

# $INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data/" \
#                                     --comment "vanilla_0.05" \
#                                     --output_dir "./Bart_Program/saves" \
#                                     --save_dir "./Bart_Program/logs" \
#                                     --sample 0.05 \
#                                     --num_train_epochs 30 \
#                                     --device cuda \
#                                     --model_name_or_path "./Bart_Program/pretrained_model" &
      
$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data_prompt/" \
                                    --comment "vanilla_prompt" \
                                    --output_dir "./Bart_Program/saves" \
                                    --save_dir "./Bart_Program/logs" \
                                    --num_train_epochs 30 \
                                    --device cuda:0 \
                                    --model_name_or_path "./Bart_Program/pretrained_model" &

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data_prompt/" \
                                    --comment "vanilla_prompt_init" \
                                    --output_dir "./Bart_Program/saves" \
                                    --save_dir "./Bart_Program/logs" \
                                    --num_train_epochs 30 \
                                    --device cuda:3 \
                                    --model_name_or_path "Bart_Program/saves/prompt_rel/epoch_27" &

wait