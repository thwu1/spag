export PYTHONPATH=.

python tools/assign_rewards.py \
    --input_data_path data/result/im_llama2_sampling_all_words.json \
    --output_data_path data/train_spag_data_im_llama2.json \
    --sft_data_path data/alpaca_train.json