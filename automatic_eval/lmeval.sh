#!/bin/bash


model=$1

lm_eval --model hf \
    --model_args pretrained=$model,dtype=bfloat16,attn_implementation="flash_attention_2" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 8 \
    --verbosity WARNING 


lm_eval --model hf \
    --model_args pretrained=$model,dtype=bfloat16,attn_implementation="flash_attention_2" \
    --tasks ai2_arc \
    --num_fewshot 25 \
    --batch_size 4 \
    --verbosity WARNING \


lm_eval --model hf \
    --model_args pretrained=$model,dtype=bfloat16,attn_implementation="flash_attention_2" \
    --tasks truthfulqa_mc2 \
    --num_fewshot 0 \
    --batch_size 8 \
    --verbosity WARNING \