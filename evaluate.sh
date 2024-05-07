lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks  mmlu \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size auto:8 \
    --output_path /scratch/tianhao/spag/llama-2-7b-eval \
    --log_samples \
    --limit 100000
