# output="./Bart_Program/processed_data_prompt_cbr/func_rel_k5"
# python -m Bart_Program.preprocess --input_dir "./dataset" \
#                                 --type 'prompt_cbr' \
#                                 --cbr_k 5 \
#                                 --recall_index_dir "Bart_Program/recall_dump/rule_func_rel" \
#                                 --output_dir $output \
#                                 --model_name_or_path "./Bart_Program/pretrained_model"
# cp ./dataset/kb.json $output