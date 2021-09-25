python -m Bart_Program.preprocess --input_dir "./dataset" \
                                --output_dir "./Bart_Program/processed_data" \
                                --model_name_or_path "./Bart_Program/pretrained_model"
cp ./dataset/kb.json ./Bart_Program/processed_data