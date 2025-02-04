#!/bin/bash
#SBATCH --job-name="evaluate"
#SBATCH  --account=bckr-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=8:59:00
#SBATCH --output=$OUTPUT
#SBATCH --error=$ERROR


# conda activate eval

echo $(which python)

model_and_tok=$MODEL_DIR

export OUTLINES_CACHE_DIR=$model_and_tok

model_str=$(basename "$model_and_tok")
echo "Evaluating model: $model_str"
echo "Model directory: $model_and_tok"

cd ./Building-Math-Agents-with-Multi-Turn-Iterative-Preference-Learning/inference/ && \
    bash scripts/register_server_single_sbatch.sh $model_and_tok && \
    sleep 60 && \
    bash scripts/infer_single.sh llama3_${model_str} \

echo "Evaluation on MATH and GSM8K completed for model: $model_str"

pkill -f "python -m vllm.entrypoints.api_server" && sleep 30

python3 -m evalplus.evaluate --model $model_and_tok --dataset mbpp --backend vllm --root ~/eval_res1/ --greedy

echo "Evaluation on MBPP completed for model: $model_str"

python3 -m evalplus.evaluate --model $model_and_tok --dataset humaneval --backend vllm --root ~/eval_res1/ --greedy

echo "Evaluation on humaneval completed for model: $model_str"

bash ./lmeval.sh $model_and_tok

echo "Evaluation on commensense completed for model: $model_str"