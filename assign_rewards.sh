MODEL_NAME=$1

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate test

PYTHONPATH=. python tools/assign_rewards.py \
    --input_data_path /scratch/tianhao/spag/data/result/${MODEL_NAME}_sampling_all_words.json \
    --output_data_path data/train_spag_data_${MODEL_NAME}.json \
    --sft_data_path data/alpaca_train.json