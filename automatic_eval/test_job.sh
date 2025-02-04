#!/bin/bash
#SBATCH --job-name="evaluate"
#SBATCH  --account=bckr-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60g
#SBATCH --time=5:59:00
#SBATCH --output="eval.log"
#SBATCH --error="eval.err"

# conda init
# conda activate eval

echo $(which python)

model_and_tok=$MODEL_DIR

model_str=$(basename "$model_and_tok")

# cd /u/xpan2/Building-Math-Agents-with-Multi-Turn-Iterative-Preference-Learning_bkp/inference/  \
#     bash scripts/register_server_single.sh $model_and_tok \
#     sleep 60 \
#     bash scripts/infer_single.sh llama3_${model_str} \

# echo "Evaluation on MATH and GSM8K completed for model: $model_str"

# pkill -f "python -m vllm.entrypoints.api_server"

# python3 -m evalplus.evaluate --model $model_and_tok --dataset mbpp --backend vllm --root ~/eval_res1/ --greedy

# echo "Evaluation on MBPP completed for model: $model_str"

# python3 -m evalplus.evaluate --model $model_and_tok --dataset humaneval --backend vllm --root ~/eval_res1/ --greedy

# echo "Evaluation on humaneval completed for model: $model_str"

# ./lmeval.sh $model_and_tok

# echo "Evaluation on commensense completed for model: $model_str"

echo $model_and_tok
echo $model_str