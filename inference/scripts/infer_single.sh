if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi
MODEL_NAME_OR_PATH=$1

#DATA_LIST = ['math', 'gsm8k']

DATA_NAME="math"

OUTPUT_DIR="~/test_collect_data"

DATA_NAME="math"
SPLIT="test"
PROMPT_TYPE="cot"
NUM_TEST_SAMPLE=-1


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.infer_eval_test \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 1 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 6 \
--ports "8000" \
--eval True \


DATA_NAME="gsm8k"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer_data.infer_eval_test \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 1 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
--horizon 6 \
--ports "8000" \
--eval True \
