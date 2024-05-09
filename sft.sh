MODEL=$1

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate test

OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=6000 train.py \
    --output_dir "./ckpts/im" \
    --model_name_or_path $MODEL \
    --ref_model_name_or_path $MODEL \
    --lm_kl_coeff 0.1 \
    --train_method "SFTwithKL" \
    --train_data_path "./data/train_imitation_gpt4.json" \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --padding_side "right" \
    --truncation_side "left" \
    --max_length 2048 \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --weight_decay 0. \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --tf32 True \
    --bf16 True