"""
Prompt strategy = "Top-K"
Sampling strategy: can be "misleading" or "self_random"; can query only once by setting num_ensemble to 1
1. read the question and true answer pair  
2. construct prompt
3. query the LLMs
4. record the results
"""
import os, pdb, time, re
import os.path as osp
import random
import json
from utils.dataset_loader import load_dataset
# from utils.llm_query_helper import calculate_result_per_question
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
import os

openai_key = "sk-Faak8mmqNrE3ChiZ323cF7Ed69D74f54A33905954dDfCfE7" # TODO: replace with your openai key

time_stamp = time.strftime("%Y-%m-%d_%H-%M")


################# CONFIG #####################
parser = ArgumentParser()
# use_cot = False
# dataset_name = "ScienceQA"
# model_name = "GPT4"
# output_dir = "output/ScienceQA/raw_results_input"
# task_type = "multi_choice_qa"
# data_path = "dataset/ScienceQA/data/scienceqa"

parser.add_argument("--dataset_name", type=str, default="hybrid_multi_choice_dataset")
parser.add_argument("--data_path", type=str, default="dataset/hybrid_multi_choice_dataset.json")
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--use_cot", action='store_true', help="default false; use cot when specified with --use_cot")
parser.add_argument("--model_name", type=str, default="gpt4")
parser.add_argument("--task_type", type=str, default="multi_choice_qa")



# for top-k prompt strategy
parser.add_argument("--prompt_type", type=str, default="top_k")
parser.add_argument("--from_peft_checkpoint", type=str, default=None)
parser.add_argument("--num_K", type=int, required=True, help="number of top K results")

# for ensemble-based methods
parser.add_argument("--sampling_type", type=str, default="misleading") # misleading or inner randomness 
parser.add_argument("--num_ensemble", type=int, default=1, required=True) 
parser.add_argument("--temperature_for_ensemble", type=float, default=0.0) # temperature for ensemble-based methods


args = parser.parse_args()



# print the args to the console
# pretty print the dictionary
print("basic information for the experiment: ")
print(json.dumps(vars(args), indent=4))

# file path
os.makedirs(osp.dirname(args.output_file), exist_ok=True)
print("output_file: ", args.output_file, "\n")

######################## READ DATASET ############################
# qa_data is the dictionary that stores the questions and answers
# questions are the keys and answers are the values
print(f"Loading dataset {args.dataset_name} from {args.data_path}")
qa_data = load_dataset(args.dataset_name, args.task_type, args.data_path)
###################### PROMPT ############################
k = args.num_K
print(f"***Using top {k} results for {args.dataset_name} {args.task_type} {args.model_name}***")

if args.task_type == "multi_choice_qa":
    task_output_description = "option letter"
elif args.task_type == "open_number_qa":
    task_output_description = "numerical answer"
elif args.task_type == "open_ended":
    task_output_description = "answer"
else:
    raise ValueError(f"task_type {args.task_type} not supported")

    
if not args.use_cot:
    prompt_description = f"Provide your {k} best guesses and the probability that each is correct (0% to 100%) for the following question. Give ONLY the {task_output_description} of your guesses and probabilities, no other words or explanation. For example:\n"
elif args.use_cot:
    prompt_description = f"Provide your {k} best guesses and the probability that each is correct (0% to 100%) for the following question. Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\n"
    

# we finally use the confidence before all answers prompting 
confidence_before_all_answers = True
if confidence_before_all_answers: 
    prompt_description += f"G1: <ONLY the {task_output_description} of first most likely guess; not a complete sentence, just the guess!>\nP1: <ONLY the probability that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nG{k}: <ONLY the {task_output_description} of {k}-th most likely guess>\nP{k}: <ONLY the probability that G{k} is correct, without any extra commentary whatsoever; just the probability!>\n"
else:
    prompt_description += f"G1: <ONLY the {task_output_description} of first most likely guess; not a complete sentence, just the guess!>\n...\nG{k}: <ONLY the {task_output_description} of {k}-th most likely guess>\n\nP1: <ONLY the probability that G1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nP{k}: <ONLY the probability that G{k} is correct, without any extra commentary whatsoever; just the probability!>\n"
    
    
hint_prompts = {
'hint0' : "",
'hint1' : "Hint: I think the answer should be ",
'hint2' : "Hint: My teacher tells me the answer is ",
'hint3' : "Hint: The textbook shows that answer is ",
'hint4' : "Hint: An expert says that the answer is ",


'hint5' : "Hint: Latest research shows that the answer is ",
'hint6' : "Hint: Wikipedia claims that the answer is ",
'hint7' : "Hint: I read online the answer is ",


'hint8' : "Hint: I vaguely remember the answer is ",
'hint9' : "Hint: I am 80% sure the answer is ",
'hint10' : "Hint: I am pretty sure the answer is ",
'hint11' : "Hint: Undoubtedly the answer is ",
'hint12' : "Hint: With my 100% confidence the answer is ",

}


############ FUNCTION DEFINITION #####################
def generate_misleading_hint(hint_type, task_type, question, qa_dataset):
    """
    1. generate misleading answers for the question
    2. task_type: open_number_qa or multi_choice_qa
    3. hint_type: hint0, hint1, hint2, hint3, hint4, hint5, hint6, hint7, hint8, hint9, hint10, hint11, hint12 (differnt levels of misleadingness)  
    """
    answer_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
    if task_type == 'open_number_qa':
        correct_answer = abs(int(qa_dataset[question]['answer']))
        noise = random.randint(-correct_answer, correct_answer)
        random_answer = correct_answer + noise 
    elif task_type == "multi_choice_qa":
        random_answer = random.randint(0, len(qa_dataset[question]['options'])-1)
        random_answer = answer_list[random_answer]
    elif task_type == 'open_ended':
        correct_answer = qa_dataset[question]['answer']
    else:
        raise ValueError(f"{task_type} not supported")
    
    if hint_type == 'hint0':
        hint_prompt = hint_prompts[hint_type]
    else:
        hint_prompt = hint_prompts[hint_type] + str(random_answer)
    
    return hint_prompt

def generate_prompt(description, question, misleading_hint, tokenizer):
    """
    1. description: prompt + answer format 
    2. the question
    3. misleading_hint
    """
    hint_description = ""
    # TODO：
    # prompt = f"{description}\n{hint_description}\nQuestion: {question}\n{misleading_hint}"
    # prompt = f"Question: {question}\n{description}\n{hint_description}\n{misleading_hint}"
    prompt = [{'role': 'system', 'content': f"{description}\n{hint_description}"},
            {"role": "user", "content":  f"Question: {question}"},
            {"role": "assistant", "content": f"Response:"},
            ]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True, max_length=8192)

    return prompt

############## MAIN FUNCTION #################### 


if args.model_name == "llama3.1":
    model_name = "../meta-llama/Meta-Llama-3.1-8B"
    model = LLM(model=model_name, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side='left'
    )
elif args.model_name == "llama3.1-instruct":
    model_name = "../meta-llama/Llama-3.1-8B-Instruct"
    model = LLM(model=model_name, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side='left'
    )
elif args.model_name == "llama3":
    model_name = "/home/lyb/workspace/llama3/llama3-8b"
    model = LLM(model=model_name, tensor_parallel_size=1)



# print sample data with sample prompt
sample_question = next(iter(qa_data))
# sample_hint_prompt = generate_misleading_hint(hint_type="hint1" if args.num_ensemble > 1 else "hint0", task_type=args.task_type, question=sample_question, qa_dataset=qa_data)
sample_prompt = generate_prompt(prompt_description, question=sample_question, misleading_hint="", tokenizer=tokenizer)
print("\n-------\n", sample_prompt, "\n-------\n")

# pdb.set_trace()

# construct the answer sheet
if os.path.exists(args.output_file):
    with open(args.output_file,'r') as f:
        final_result = json.load(f).get("final_result", {})
else:
    final_result = {}

# save parameters
params = vars(args)

# tell if the output file exists
if args.output_file is None:
    args.output_file = f"final_output/{args.prompt_type}_{args.sampling_type}/" + args.model_name + "/" + args.dataset_name + "/" + f"{args.dataset_name}_{args.model_name}_{time_stamp}.json"
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
print("output_file: ", args.output_file, "\n")



# record time
start_time = time.time()

error_dataset = {}


# Collect all prompts first
all_prompts = []
all_questions = []
all_hint_types = []

for idx, question in enumerate(qa_data.keys()):
    if question in final_result:
        print(f"Question: [{question}] already in final_result, skip")
        continue
    
    final_result[question] = {}
    if args.sampling_type == "misleading":
        test_hints = ["hint0"] + ["hint" + str(i) for i in range(1, args.num_ensemble)]
        assert len(test_hints) <= len(hint_prompts), f"number of hints {len(test_hints)} should be less than or equal to {len(hint_prompts)}; otherwise we need to add more hint prompts"
        for hint_type in test_hints:
            misleading_hint = generate_misleading_hint(hint_type, args.task_type, question, qa_data)
            prompt = generate_prompt(prompt_description, question, misleading_hint, tokenizer)
            print(f"Collecting prompt for {question} with {hint_type}")
            
            all_prompts.append(prompt)
            all_questions.append(question)
            all_hint_types.append(hint_type)
            
    elif args.sampling_type == "self_random":
        for ith in range(args.num_ensemble):
            prompt = generate_prompt(prompt_description, question, misleading_hint="", tokenizer=tokenizer)
            hint_type = f"trail_{ith}"
            
            all_prompts.append(prompt)
            all_questions.append(question)
            all_hint_types.append(hint_type)

# Generate all responses at once using vLLM
if args.model_name.lower() in ['llama3.1', 'llama3.1-instruct', 'llama3']:
    print(f"\nGenerating responses for {len(all_prompts)} prompts...")
    max_tokens = 2000 if args.use_cot else 400
    sampling_params = SamplingParams(
        temperature=args.temperature_for_ensemble,
        max_tokens=max_tokens
    )
    outputs = model.generate(all_prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    
    # Process responses and update final_result
    for question, hint_type, response in zip(all_questions, all_hint_types, responses):
        if hint_type.startswith('hint'):
            final_result[question][hint_type] = {
                'hint_response': response,
                'real_answer': qa_data[question],
                'hint_entry': hint_prompts[hint_type] + (str(generate_misleading_hint(hint_type, args.task_type, question, qa_data)) if hint_type != 'hint0' else '')
            }
        else:  # self_random case
            final_result[question][hint_type] = {
                'hint_response': response,
                'real_answer': qa_data[question]
            }
        
        # print(f"\nQuestion: {question[:100]}...")
        # print(f"Response ({hint_type}): {response}")
        # print("-" * 70)
        
        # Save results periodically
        if len(final_result) % 5 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            final_json = {
                'hyperparameters': params,
                'elapsed_time': f"Elapsed time: {elapsed_time:.2f} seconds",
                'sample_tested': len(final_result),
                'error_count': len(error_dataset),
                'sample_prompt': {
                    'question': sample_question,
                    'hint': "",
                    'prompt': sample_prompt
                },
                'final_result': final_result,
                'error_dataset': error_dataset
            }
            with open(args.output_file, 'w') as f:
                f.write(json.dumps(final_json, indent=4))

else:
    # For non-vLLM models, use the original processing method
    for idx, (question, prompt, hint_type) in enumerate(zip(all_questions, all_prompts, all_hint_types)):
        final_result, error_dataset = calculate_result_per_question(
            args.model_name, question, prompt,
            final_result, error_dataset, qa_data,
            hint_type, args.task_type, args.use_cot,
            openai_key=openai_key,
            temperature=args.temperature_for_ensemble
        )

# Save final results
end_time = time.time()
elapsed_time = end_time - start_time
final_json = {
    'hyperparameters': params,
    'elapsed_time': f"Elapsed time: {elapsed_time:.2f} seconds",
    'sample_tested': len(final_result),
    'error_count': len(error_dataset),
    'sample_prompt': {
        'question': sample_question,
        'hint': "",
        'prompt': sample_prompt
    },
    'final_result': final_result,
    'error_dataset': error_dataset
}
with open(args.output_file, 'w') as f:
    f.write(json.dumps(final_json, indent=4))