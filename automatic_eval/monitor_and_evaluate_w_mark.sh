#!/bin/bash

WATCH_DIR="/scratch/bckr/xpan2/projects/rlhflow_models"
TEMPLATE="/u/xpan2/automatic_eval/eval_job_w_mark.sh"
CHECK_INTERVAL=10  # Check every 60 seconds

while true; do
    # Find all directories with "model_ready" files
    for READY_FILE in $(find "$WATCH_DIR" -name "model_ready"); do
        MODEL_DIR=$(dirname "$READY_FILE")
        MODEL_NAME=$(basename "$MODEL_DIR")

        # Check if all evaluation steps are completed
        math_mark="${MODEL_DIR}/math_completed.mark"
        mbpp_mark="${MODEL_DIR}/mbpp_evaluation_completed.mark"
        humaneval_mark="${MODEL_DIR}/humaneval_evaluation_completed.mark"
        commonsense_mark="${MODEL_DIR}/commonsense_evaluation_completed.mark"

        # If all marks exist, skip this model
        if [[ -f "$math_mark" && -f "$mbpp_mark" && -f "$humaneval_mark" && -f "$commonsense_mark" ]]; then
            echo "All evaluations completed for model: $MODEL_NAME. Skipping."
            continue
        fi

        in_progress_mark="${MODEL_DIR}/evaluation_in_progress.mark"
        if [[ -f "$in_progress_mark" ]]; then
            echo "Evaluation in progress for model: $MODEL_NAME. Waiting for completion."
            continue
        fi

        echo "Starting evaluation for model: $MODEL_NAME..."
        touch "$in_progress_mark"
        # Create a temporary SLURM job script from the template
        TEMP_JOB_SCRIPT="$MODEL_DIR/evaluate_${MODEL_NAME}.sbatch"
        sed "s|\$MODEL_DIR|$MODEL_DIR|g" "$TEMPLATE" | \
            sed "s|\$OUTPUT|$MODEL_DIR/${MODEL_NAME}_%j.out|g" | \
            sed "s|\$ERROR|$MODEL_DIR/${MODEL_NAME}_%j.err|g" \
            > "$TEMP_JOB_SCRIPT"
        
        
        sbatch "$TEMP_JOB_SCRIPT"

        # Re-check if all marks exist after running the evaluation script
        # if [[ -f "$math_mark" && -f "$mbpp_mark" && -f "$humaneval_mark" && -f "$commonsense_mark" ]]; then
        #     echo "All evaluations completed successfully for model: $MODEL_NAME."
        # else
        #     echo "Evaluation for model: $MODEL_NAME did not complete. It will retry in the next cycle."
        # fi
    done

    sleep "$CHECK_INTERVAL"
done