#!/bin/bash

PROMPT_TYPE="top_k"
SAMPLING_TYPE="self_random" 
NUM_ENSEMBLE=3
CONFIDENCE_TYPE="${PROMPT_TYPE}_${SAMPLING_TYPE}_${NUM_ENSEMBLE}"
# directory of pre-trained peft model
# PEFT="/home/lyb/workspace/llama-recipes/src/checkpoints/professional1106_brier_log"

#############################################################

# DATASET_NAME="Professional_Law"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="/home/lyb/workspace/dataset/data/val/professional_law_val.csv"
# USE_COT=false # use cot or not
# TEMPERATURE=0.7
# TOP_K=2

#DATASET_NAME="BigBench_ObjectCounting"
#MODEL_NAME="llama3.1-instruct"
#TASK_TYPE="open_number_qa"
#DATASET_PATH="/home/lyb/workspace/dataset/ObjectCou/task.json"
#USE_COT=false # use cot or not
#TEMPERATURE=0.7
#TOP_K=2


DATASET_NAME="GSM8K"
MODEL_NAME="gpt4o"
TASK_TYPE="open_number_qa"
DATASET_PATH="dataset/grade_school_math/data/test.jsonl"
USE_COT=true # use cot or not
TEMPERATURE=0
TOP_K=3


# DATASET_NAME="trivia_qa"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="open_ended"
# DATASET_PATH="dataset"
# USE_COT=true # use cot or not
# TEMPERATURE=0.1
# TOP_K=3


# DATASET_NAME="hotpot_qa"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="open_ended"
# DATASET_PATH="dataset"
# USE_COT=true # use cot or not
# TEMPERATURE=0.1
# TOP_K=3

# DATASET_NAME="truthful_qa"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="open_ended"
# DATASET_PATH="dataset"
# USE_COT=true # use cot or not
# TEMPERATURE=0.1
# TOP_K=3

# DATASET_NAME="BigBench_DateUnderstanding"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/BigBench/date_understanding.json"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4


# DATASET_NAME="Business_Ethics"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4

# DATASET_NAME="BigBench_strategyQA"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/StrategyQA/task.json"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4

# DATASET_NAME="BigBench_sportUND"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4

#############################################################
# set time stamp to differentiate the output file
TIME_STAMPE=$(date "+%m-%d-%H-%M")

OUTPUT_DIR="final_output/$CONFIDENCE_TYPE/$MODEL_NAME/$DATASET_NAME"
RESULT_FILE="$OUTPUT_DIR/${DATASET_NAME}_${MODEL_NAME}_${TIME_STAMPE}.json"
USE_COT_FLAG=""

if [ "$USE_COT" = true ] ; then
    USE_COT_FLAG="--use_cot"
fi

set -x
python query_top_k.py \
   --dataset_name  $DATASET_NAME \
   --data_path $DATASET_PATH \
   --output_file  $RESULT_FILE \
   --model_name  $MODEL_NAME \
   --task_type  $TASK_TYPE  \
   --prompt_type $PROMPT_TYPE \
   --num_K $TOP_K \
   --sampling_type $SAMPLING_TYPE \
   --num_ensemble $NUM_ENSEMBLE \
   --temperature_for_ensemble $TEMPERATURE \
   $USE_COT_FLAG


# uncomment following lines to run test and visualization
python extract_answers.py \
   --input_file $RESULT_FILE \
   --model_name  $MODEL_NAME \
   --dataset_name  $DATASET_NAME \
   --task_type  $TASK_TYPE   \
   --prompt_type $PROMPT_TYPE \
   --num_K $TOP_K \
   --sampling_type $SAMPLING_TYPE \
   --num_ensemble $NUM_ENSEMBLE \
    $USE_COT_FLAG

RESULT_FILE_PROCESSED=$(echo $RESULT_FILE | sed 's/\.json$/_processed.json/')

python vis_aggregated_conf_top_k.py \
    --input_file $RESULT_FILE_PROCESSED \
    --model_name  $MODEL_NAME \
    --dataset_name  $DATASET_NAME \
    --task_type  $TASK_TYPE   \
    --prompt_type $PROMPT_TYPE  \
    --num_K $TOP_K \
    --sampling_type $SAMPLING_TYPE \
    --num_ensemble $NUM_ENSEMBLE \
    $USE_COT_FLAG    


