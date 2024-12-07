#!/bin/bash

# SAMPLING_TYPE="self_random" 
# NUM_ENSEMBLE=1
# CONFIDENCE_TYPE="${SAMPLING_TYPE}_${NUM_ENSEMBLE}"
# PEFT="/home/lyb/workspace/Uncertainty_ft/src/checkpoints/object_1103_llm2"


# TODO uncomment following lines to run on different settings
#############################################################

DATASET_NAME="Professional_Law"
MODEL_NAME="llama3.1-instruct"
TASK_TYPE="multi_choice_qa"
DATASET_PATH="/home/lyb/workspace/Uncertainty_ft/dataset/data/val/professional_law_val.csv"
# USE_COT=true # use cot or not
TEMPERATURE=0.7

# DATASET_NAME="BigBench_ObjectCounting"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="open_number_qa"
# DATASET_PATH="/home/lyb/workspace/Uncertainty_ft/dataset/ObjectCou/task.json"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=2

#DATASET_NAME="BigBench_ObjectCounting"
#MODEL_NAME="llama3.1-instruct"
#TASK_TYPE="open_number_qa"
#DATASET_PATH="/home/lyb/workspace/dataset/ObjectCou/task.json"
#USE_COT=true # use cot or not
#TEMPERATURE=0.7
#TOP_K=2


# DATASET_NAME="GSM8K"
# MODEL_NAME="llama3.1-instruct"
# TASK_TYPE="open_number_qa"
# DATASET_PATH="/home/lyb/workspace/dataset/grade_school_math/data/test.jsonl"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=2
 #TIME_STAMPE="09-14-01-06"

# DATASET_NAME="BigBench_DateUnderstanding"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/BigBench/date_understanding.json"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4
# TIME_STAMPE="09-14-01-30"


# DATASET_NAME="Business_Ethics"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4
# TIME_STAMPE="09-23-13-44"

# DATASET_NAME="BigBench_strategyQA"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4
# TIME_STAMPE="09-23-07-42"

# DATASET_NAME="BigBench_sportUND"
# MODEL_NAME="gpt4"
# TASK_TYPE="multi_choice_qa"
# DATASET_PATH="dataset/MMLU/business_ethics_test.csv"
# USE_COT=true # use cot or not
# TEMPERATURE=0.7
# TOP_K=4
# TIME_STAMPE="09-23-05-28"

#############################################################
# set time stamp to differentiate the output file
TIME_STAMPE=$(date "+%m-%d-%H-%M")

OUTPUT_DIR="final_output/generate_answer/$MODEL_NAME/$DATASET_NAME"
RESULT_FILE="$OUTPUT_DIR/${DATASET_NAME}_${MODEL_NAME}_${TIME_STAMPE}.json"
USE_COT_FLAG=""

if [ "$USE_COT" = true ] ; then
    USE_COT_FLAG="--use_cot"
fi

set -x
python query_top_answer.py \
   --dataset_name  $DATASET_NAME \
   --data_path $DATASET_PATH \
   --output_file  $RESULT_FILE \
   --model_name  $MODEL_NAME \
   --task_type  $TASK_TYPE  \
   --temperature_for_ensemble $TEMPERATURE \
   #--from_peft_checkpoint $PEFT \
   $USE_COT_FLAG
 


