#!/bin/bash

WATCH_DIR="/scratch/bckr/xpan2/projects/rlhflow_models"
TEMPLATE="/u/xpan2/automatic_eval/eval_job.sh"
CHECK_INTERVAL=10  # Check every 60 seconds

while true; do
    # Find all "model_ready" files in the shared directory
    for READY_FILE in $(find "$WATCH_DIR" -name "model_ready"); do
        MODEL_DIR=$(dirname "$READY_FILE")
        MODEL_NAME=$(basename "$MODEL_DIR")
        # echo $MODEL_NAME
        # echo $READY_FILE
        # echo $MODEL_NAME
        # echo $MODEL_DIR

        # Check if evaluation has already been triggered
        if [[ -f "$MODEL_DIR/evaluation_started" ]]; then
            continue
        fi

        echo "Detected new model: $MODEL_NAME. Starting evaluation..." | tee /dev/tty
        # fflush
        # Create a temporary SLURM job script from the template
        TEMP_JOB_SCRIPT="$MODEL_DIR/evaluate_${MODEL_NAME}.sbatch"
        sed "s|\$MODEL_DIR|$MODEL_DIR|g" "$TEMPLATE" | \
            sed "s|\$OUTPUT|$MODEL_DIR/${MODEL_NAME}_%j.out|g" | \
            sed "s|\$ERROR|$MODEL_DIR/${MODEL_NAME}_%j.err|g" \
            > "$TEMP_JOB_SCRIPT"
        
        
        sbatch "$TEMP_JOB_SCRIPT"

        # Mark the model as being evaluated
        touch "$MODEL_DIR/evaluation_started"
    done

    sleep "$CHECK_INTERVAL"
done