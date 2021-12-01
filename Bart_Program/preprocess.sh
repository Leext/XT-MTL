# output="./Bart_Program/processed_data_prompt_cbr/func_rel_k5"
# python -m Bart_Program.preprocess --input_dir "./dataset" \
#                                 --type 'prompt_cbr' \
#                                 --cbr_k 5 \
#                                 --recall_index_dir "Bart_Program/recall_dump/rule_func_rel" \
#                                 --output_dir $output \
#                                 --model_name_or_path "./Bart_Program/pretrained_model"
# cp ./dataset/kb.json $output

INSPUR1="srun -p inspur -w inspur-gpu-05" 
# INSPUR1="srun --gres=gpu:V100:1"
function preprocess_cbr() {
    sample=$1
    dataset_path="./dataset_$sample"
    mkdir $dataset_path
    cp ./dataset/* $dataset_path
    python ./Bart_Program/sample.py ./dataset/train.json $dataset_path/train.json $sample
    output="./Bart_Program/processed_data_$sample"
    python -m Bart_Program.preprocess --input_dir $dataset_path \
                                --type 'default' \
                                --output_dir $output \
                                --model_name_or_path "./Bart_Program/pretrained_model"
    recall_output="./Bart_Program/recall_dump/rule_func_rel_$sample"
    $INSPUR1 python -m Bart_Program.recall --input_dir $output \
                        --save_dir "./Bart_Program/logs" \
                        --model_name_or_path "./Bart_Program/pretrained_model" \
                        --ckpt "Bart_Program/saves/recall_rule_rel/epoch_3" \
                        --rep_fn "func_rel_seq" \
                        --simi_fn "edit" \
                        --device cuda:1 \
                        --recall_dump $recall_output
    cbr_output="./Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"
    
    python -m Bart_Program.preprocess --input_dir $dataset_path \
                                --type 'prompt_cbr' \
                                --cbr_k 5 \
                                --recall_index_dir $recall_output \
                                --output_dir $cbr_output \
                                --model_name_or_path "./Bart_Program/pretrained_model"
}

function preprocess_cbr_boost() {
    sample=$1
    dataset_path="./dataset_$sample"
    # mkdir $dataset_path
    # cp ./dataset/* $dataset_path
    # python ./Bart_Program/sample.py ./dataset/train.json $dataset_path/train.json $sample
    output="./Bart_Program/processed_data_$sample"
    python -m Bart_Program.preprocess --input_dir $dataset_path \
                                --type 'default' \
                                --output_dir $output \
                                --model_name_or_path "./Bart_Program/pretrained_model"

    recall_output="./Bart_Program/recall_dump/rule_func_rel_$sample"
    $INSPUR1 python -m Bart_Program.recall --input_dir $output \
                        --save_dir "./Bart_Program/logs" \
                        --model_name_or_path "./Bart_Program/pretrained_model" \
                        --ckpt "Bart_Program/saves/recall_rule_rel/epoch_3" \
                        --rep_fn "func_rel_seq" \
                        --simi_fn "edit" \
                        --device cuda:1 \
                        --recall_dump $recall_output

    cbr_output="./Bart_Program/processed_data_prompt_cbr/func_rel_k5_boost_$sample" 
    python -m Bart_Program.preprocess --input_dir $dataset_path \
                                --type 'prompt_cbr' \
                                --cbr_k 5 \
                                --boost \
                                --recall_index_dir $recall_output \
                                --output_dir $cbr_output \
                                --model_name_or_path "./Bart_Program/pretrained_model"
}

preprocess_cbr_boost 100
preprocess_cbr_boost 200
preprocess_cbr_boost 500
# sample="rel_exp"
# dataset_path='dataset_rel'
# output="./Bart_Program/processed_data_rel_exp"
# python -m Bart_Program.preprocess --input_dir $dataset_path \
#                             --type 'default' \
#                             --output_dir $output \
#                             --model_name_or_path "./Bart_Program/pretrained_model"
# recall_output="./Bart_Program/recall_dump/rule_func_rel_$sample"
# $INSPUR1 python -m Bart_Program.recall --input_dir $output \
#                     --save_dir "./Bart_Program/logs" \
#                     --model_name_or_path "./Bart_Program/pretrained_model" \
#                     --ckpt "Bart_Program/saves/recall_rule_rel/epoch_3" \
#                     --rep_fn "func_rel_seq" \
#                     --simi_fn "edit" \
#                     --device cuda \
#                     --recall_dump $recall_output
# cbr_output="./Bart_Program/processed_data_prompt_cbr/func_rel_k5_$sample"

# python -m Bart_Program.preprocess --input_dir $dataset_path \
#                             --type 'prompt_cbr' \
#                             --cbr_k 5 \
#                             --recall_index_dir $recall_output \
#                             --output_dir $cbr_output \
#                             --model_name_or_path "./Bart_Program/pretrained_model"