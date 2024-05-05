export PYTHONPATH=.

python tools/play_llm_game_api.py \
    --taboo_max_turns 5 \
    --attacker_model_name_or_path "./ckpt" \
    --defender_model_name_or_path "./ckpt" \
    --model_prefix "im_llama2" \
    --data_path "./data/all_target_words.txt" \
    --output_dir "./data/result" \
    --per_device_eval_batch_size 1 \
    --task_type "sampling" \
    --data_suffix "all_words" \
    --max_length 2048 \
    --max_new_tokens 256 \
    --logging_steps 5 \
    --batch_size 10000 \
    --bf16 True \
    --tf32 True