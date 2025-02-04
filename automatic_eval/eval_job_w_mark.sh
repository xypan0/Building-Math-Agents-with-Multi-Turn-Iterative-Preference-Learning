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

echo $(which python)

model_and_tok=$MODEL_DIR
export OUTLINES_CACHE_DIR=$model_and_tok

model_str=$(basename "$model_and_tok")
echo "Evaluating model: $model_str"
echo "Model directory: $model_and_tok"


math_mark="${model_and_tok}/math_completed.mark"
if [[ ! -f "$math_mark" ]]; then
    echo "evaluating math for model: $model_str..."
    cd /u/xpan2/Building-Math-Agents-with-Multi-Turn-Iterative-Preference-Learning/inference/ && \
    bash scripts/register_server_single_sbatch.sh "$model_and_tok" && sleep 60 && \
    bash scripts/infer_single.sh "llama3_${model_str}" && \
    touch "$math_mark" && \
    echo "math evaluation completed."
    pkill -f "python -m vllm.entrypoints.api_server" && sleep 30
else
    echo "math evaluation already completed. Skipping."
fi


# Evaluation on MBPP
mbpp_mark="${model_and_tok}/mbpp_evaluation_completed.mark"
if [[ ! -f "$mbpp_mark" ]]; then
    echo "Evaluating MBPP for model: $model_str..."
    python3 -m evalplus.evaluate --model "$model_and_tok" --dataset mbpp --backend vllm --root ~/eval_res1/ --greedy && \
    touch "$mbpp_mark"
    echo "MBPP evaluation completed."
else
    echo "MBPP evaluation already completed. Skipping."
fi

# Evaluation on HumanEval
humaneval_mark="${model_and_tok}/humaneval_evaluation_completed.mark"
if [[ ! -f "$humaneval_mark" ]]; then
    echo "Evaluating HumanEval for model: $model_str..."
    python3 -m evalplus.evaluate --model "$model_and_tok" --dataset humaneval --backend vllm --root ~/eval_res1/ --greedy && \
    touch "$humaneval_mark"
    echo "HumanEval evaluation completed."
else
    echo "HumanEval evaluation already completed. Skipping."
fi

# Evaluation on commonsense (LMEval)
commonsense_mark="${model_and_tok}/commonsense_evaluation_completed.mark"
if [[ ! -f "$commonsense_mark" ]]; then
    echo "Evaluating commonsense tasks for model: $model_str..."
    bash /u/xpan2/automatic_eval/lmeval.sh "$model_and_tok"  && \
    touch "$commonsense_mark"
    echo "Commonsense evaluation completed."
else
    echo "Commonsense evaluation already completed. Skipping."
fi

rm ${model_and_tok}/evaluation_in_progress.mark