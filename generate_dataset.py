"""
Aggregation Strategy:
    - "AVG-Conf"
    - "Consistency"
    - "Pair-Rank"

Supported Prompt Strategy:
    - "Top-K"

This code is used to visualize the performance (e.g. ACC/AUCROC/ECE) of  **Confidence Scores based on misleading consistency**:
- ACC
- AUCROC
- AUPRC
- ECE

"""

# %%
import json, os, sys, pdb, json
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib
import datasets
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
from argparse import ArgumentParser
from adjustText import adjust_text
from collections import Counter
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
               "V", "W", "X", "Y", "Z"]

# %%
################# CONFIG #####################
parser = ArgumentParser()

# ############## GSM8K GPT3.5 ################
# model_name = "GPT3-5"
# description = "Non-COT"
# dataset_name = "GSM8K"
# input_file = "GSM8K_result_chatgpt.json"
# main_folder = "output/misleading"

# ############## ScienceQA GPT3.5 ################
# model_name = "GPT3-5"
# description = "Non-COT"
# dataset_name = "ScienceQA"
# input_file = "ScienceQA_final_result_chatgpt.json"
# main_folder = "output/misleading"

# ############## ScienceQA GPT3.5 ################
# model_name = "GPT3-5"
# description = "Non-COT"
# dataset_name = "DateUnderstanding"
# input_file = "date_understanding_final_result_chatgpt.json"
# main_folder = "output/misleading"


parser.add_argument("--input_file", type=str,
                    default="output/consistency/raw_results_input/BigBench_ObjectCounting_gpt3_2023-04-27_01-09_processed.json")
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="gpt4")
parser.add_argument("--dataset_name", type=str, default="BigBench_DateUnderstanding")
parser.add_argument("--task_type", type=str, default="multi_choice_qa")

# for prompt strategy
parser.add_argument("--prompt_type", type=str, choices=["vanilla", "cot", "self_probing", "top_k", "multi_step"],
                    default="vanilla")
parser.add_argument("--num_K", type=int, required=True, help="number of top K results")

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="misleading")  # misleading or inner randomness
parser.add_argument("--num_ensemble", type=int, default=1, help="number of queries to ensemble for a given question")

args = parser.parse_args()

main_folder = os.path.dirname(args.input_file)
input_file_name = os.path.basename(args.input_file)

################## READ DATA ####################

visual_folder = osp.join(main_folder, "visual")
log_folder = osp.join(main_folder, "log")
output_file = osp.join(main_folder, "all_results.csv")
os.makedirs(osp.join(main_folder, "log"), exist_ok=True)
os.makedirs(osp.join(main_folder, "visual"), exist_ok=True)

result_file_error_log = osp.join(log_folder, input_file_name.replace(".json", "_visual_error.log"))
visual_result_file = osp.join(visual_folder, input_file_name.replace(".json", ".png"))

# read all the json files
with open(osp.join(args.input_file), "r") as f:
    data_results = json.load(f)

print("data_results.keys():", data_results.keys())
data = data_results['processed_data']

# if hyperparmeters are in data, use this to replace args parameters
if 'hyperparameters' in data_results:
    assert args.model_name == data_results['hyperparameters']['model_name']
    # assert args.dataset_name == data_results['hyperparameters']['dataset_name']
    assert args.task_type == data_results['hyperparameters']['task_type']
    assert args.use_cot == data_results['hyperparameters']['use_cot'], (
    args.use_cot, data_results['hyperparameters']['use_cot'])
    assert args.prompt_type == data_results['hyperparameters']['prompt_type']
    assert args.num_K == data_results['hyperparameters']['num_K']

with open(result_file_error_log, "a") as f:
    print("sample size: ", len(data))
    f.write("sample size: " + str(len(data)) + "\n")
    # print a sample result
    for key, value in data.items():
        print("key:", key)
        for hint_key, hint_value in value.items():
            print(hint_key, ":", hint_value)
            f.write(str(hint_key) + ":" + str(hint_value) + "\n")
        break


# %%
############### EXTRA INFORMATION FROM RESULTS ####################

def compute_rank_based_top_k_score(hint_answers):
    """
    - implementation of Pair-Rank aggregation algorithm
    - rank_matrix = [[0, 1/K, ...], [],...]
        - rank_matrix[i,j]=1/K : The probability that option i appears before option j
    - see the paper for details
    """
    import torch
    import torch.optim as optim

    def convert_to_rank_matrix(answer_set):
        num_trail = len(answer_set)
        top_k = len(next(iter(answer_set.values())))

        # compute the number of unique answers
        unique_elements = set()
        for inner_dict in answer_set.values():
            unique_elements.update(inner_dict.values())
        num_options = len(unique_elements)
        # map every item to its unique id
        # element_to_id["A"]=0
        element_to_id = {element: idx for idx, element in enumerate(unique_elements)}
        id_to_element = {idx: element for element, idx in element_to_id.items()}

        rank_matrix = torch.zeros(num_options, num_options)
        for trail, answers in answer_set.items():
            # answers[trail_0] = {0:"A", ..."3":"D"}
            mask = torch.ones(num_options)
            for idx in range(top_k):
                # answer["0"] = "A" -> option
                option = answers[str(idx)]
                id_cat = element_to_id[option]
                mask[id_cat] = 0
                rank_matrix[id_cat, :] += mask

        rank_matrix = rank_matrix / num_trail
        # assert rank_matrix.any() >= 0.0 and rank_matrix.any() <=1.0, "rank matrix should be [0,1]"
        return rank_matrix, num_options, top_k, id_to_element

    rank_matrix, num_options, top_k, id_to_element = convert_to_rank_matrix(hint_answers)

    w_cat = torch.randn(num_options, requires_grad=True)

    # Define the SGD optimizer
    optimizer = optim.SGD([w_cat], lr=0.01)

    # Define the loss function: Frobenius norm of W
    def nll_loss_func(w_cat, rank_matrix):
        p_cat = torch.nn.functional.softmax(w_cat, dim=0)
        loss = 0
        # Compute the denominator for all combinations of p_cat[row] and p_cat[col]
        # denominator[i,j] = p_cat[i] + p_cat[j]
        denominator = p_cat.view(-1, 1) + p_cat.view(1, -1)
        # Avoid division by zero by adding a small constant
        epsilon = 1e-10
        # Compute the ratio
        ratios = (p_cat.view(-1, 1) + epsilon) / (denominator + 2 * epsilon)
        loss = -torch.sum(rank_matrix * ratios)
        return loss

    # Training loop to minimize the loss function

    for _ in range(1000):
        # Compute the loss
        loss = nll_loss_func(w_cat, rank_matrix)

        # Zero gradients, backward pass, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_p_cat = torch.nn.functional.softmax(w_cat, dim=0)
    id = torch.argmax(final_p_cat)
    answer = id_to_element[int(id)]
    final_p_cat = final_p_cat.detach().numpy()
    score = final_p_cat[id]

    return answer, score


if args.dataset_name in ["BigBench_DateUnderstanding"]:
    normal_option_list = ["A", "B", "C", "D", "E", "F", "G"]
elif args.dataset_name in ["Professional_Law", "Business_Ethics"]:
    normal_option_list = ["A", "B", "C", "D"]
elif args.dataset_name in ["sportUND", "strategyQA", "StrategyQA", "Bigbench_strategyQA", "BigBench_sportUND",
                           "BigBench_strategyQA"]:
    normal_option_list = ["A", "B"]
elif args.dataset_name in ["GSM8K", "BigBench_ObjectCounting"]:
    normal_option_list = None
else:
    raise NotImplementedError(f"Please specify the normal_option_list for this dataset {args.dataset_name}")


def generate(result_dict, task_type):
    """
    Function purpose:
        - This function is implemented as the aggregation strategy, i.e., used to get the final answer and confidence based on the top-k results for every question, and aggregate all ensembles to get their corresponding scores:
        - Prompt Strategy = "Top-K"

    Aggregation Strategy:
        - "AVG-Conf"
        - "Consistency"
        - "Pair-Rank"

    Hyperparameters:
        - result_dict: dict of all intermediate results
    """
    # for every question in the dataset, get their answers and their corresponding confidences -> can be multiple if num_ensemble > 1
    count = 0
    for key, value in result_dict.items():
        # if count == 0:
        #     count += 1
        #     continue
        count += 1
        """ Example below:
        - key: question text
        - value: dict of all intermediate results
            - value['hint'] (keys: 'hint_response', 'generated_result', 'real_answer')
                - value['hint']['real_answer'] = {'options': 'A', 'option_number': 6}
                - value['hint']['generated_resutl'] 
                    - (keys: 'step1', 'step2', 'step3', 'final_answer', 'final_confidence')
                - value['hint']['generated_resutl']['step1'] = {'analysis': 'xxxx Confidence: 90;', 'confidence': 90}

        """
        real_answer = value['real_answer']
        if task_type == "open_number_qa":
            real_answer = float(real_answer)

        elif task_type == 'multi_choice_qa':
            if isinstance(real_answer, int):
                real_answer = option_list[real_answer]

        # get predicted answers and confidences over multiple queries -> for ensemble
        # hint_answers = {"trai_0":{"0":"A", "1":"B"}, "trail_1":{}}
        # hint_confs = {"trai_0":{"0":90, "1":80}, "trail_1":{}}
        hint_answers = value['hint_answers']
        hint_confs = value['hint_confs']
        assert len(hint_answers) == len(hint_confs), "(len(hint_answers) should be equivalent to len(hint_confidences))"


        for trail, answers in hint_answers.items():
            if answers['0'] is None:
                continue
            elif task_type == "multi_choice_qa":
                if answers['0'] not in normal_option_list:
                    continue
                if answers['0'] == real_answer:
                    result_dict[key]['neg_llm2'] = answers['1']
                    result_dict[key]['label'] = 1
                else:
                    result_dict[key]['neg_llm2'] = answers['0']
                    result_dict[key]['label'] = 0
            else:
                if answers['0'] == real_answer:
                    result_dict[key]['neg_llm2'] = answers['1']
                    result_dict[key]['label'] = 1
                else:
                    result_dict[key]['neg_llm2'] = answers['0']
                    result_dict[key]['label'] = 0


    return result_dict


# pdb.set_trace()
result_dict = generate(data, args.task_type, )  # num_ensemble=args.num_ensemble)

# path = '/home/lyb/workspace/llama-recipes/dataset/data/test/professional_law_test_neg.csv'
# df = pd.read_csv(path)
# def insert_neg(row):
#     # 获取当前行的第一个元素作为键
#     key = row[0]
#     # 查找字典中对应的值，如果没有找到则返回 None
#     a = result_dict.get(key, None)
#     neg_llm2 = a['neg_llm2']
#     return neg_llm2
#
# def insert_label(row):
#     # 获取当前行的第一个元素作为键
#     key = row[0]
#     # 查找字典中对应的值，如果没有找到则返回 None
#     a = result_dict.get(key, None)
#     label = a['label']
#     return label
#
# # 创建新列，并通过应用函数插入字典中的值
# df['neg_llm2'] = df.apply(insert_neg, axis=1)
# df['label'] = df.apply(insert_label, axis=1)
#
# # 过滤掉 'New_Column' 为 None 的行
# df = df[df['neg_llm2'].notna()]
#
# # 保存修改后的CSV文件
# df.to_csv('/home/lyb/workspace/llama-recipes/dataset/data/test/professional_law_test_neg2.csv', index=False)
output_file = '/home/lyb/workspace/llama-recipes/dataset/ObjectCou/test_neg2.jsonl'
with open(output_file, 'w') as file:
    count = 0
    for key, value in result_dict.items():
        count += 1
        if count < 800:
            continue
        value["question"] = key
        if value.get('label') == None:
            continue
        json_line = json.dumps(value)
        file.write(json_line + '\n')




