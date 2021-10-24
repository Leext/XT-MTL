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
#                                     --comment "vanilla_predict" \
#                                     --output_dir "./Bart_Program/saves" \
#                                     --save_dir "./Bart_Program/logs" \
#                                     --valid \
#                                     --device cuda \
#                                     --model_name_or_path "Bart_Program/saves/vanilla/epoch_24" &

$INSPUR1 python -m Bart_Program.train --input_dir "./Bart_Program/processed_data/" \
                                    --comment "vanilla_predict" \
                                    --output_dir "./Bart_Program/saves" \
                                    --save_dir "./Bart_Program/logs" \
                                    --valid \
                                    --device cuda \
                                    --model_name_or_path "Bart_Program/saves/vanilla/epoch_24"
