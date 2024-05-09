MODEL=$1

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate eval

lm_eval --model hf \
    --model_args pretrained="${MODEL}" \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size auto:8 \
    --output_path /scratch/tianhao/spag/eval \
    --log_samples \
    --limit 100000
