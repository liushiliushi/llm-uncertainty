import openai, pdb
from openai import OpenAI
import requests
import json
from typing import List, Dict, Tuple, Any
#calculate_result_per_question('vicuna', 'a', 'a', {}, {}, {}, 'hint0', 'multi', False)
def process_batch(
    model_name: str,
    batch_questions: List[str],
    batch_prompts: List[str],
    batch_hint_types: List[str],
    final_result: Dict,
    error_dataset: Dict,
    qa_dataset: Dict,
    task_type: str,
    use_cot: bool,
    openai_key: str,
    temperature: float = 0.0,
    model: Any = None
) -> Tuple[Dict, Dict]:
    """
    Process a batch of questions and prompts
    """
    max_tokens = 2000 if use_cot else 400
    max_req_count = 3
    req_success = False
    
    while not req_success and max_req_count > 0:
        try:
            if model_name.lower() in ['llama3.1', 'llama3.1-instruct', 'llama3']:
                # Use vLLM batch inference
                responses = generate_with_vllm(
                    llm=model,
                    prompts=batch_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            elif model_name.lower() == 'chatgpt':
                # Process ChatGPT requests sequentially
                responses = []
                for prompt in batch_prompts:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    responses.append(response['choices'][0]['message']['content'])
            
            elif model_name.lower() == 'gpt4':
                # Process GPT-4 requests sequentially
                responses = []
                url = "https://api2.aigcbest.top/v1/chat/completions"
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {openai_key}'
                }
                
                for prompt in batch_prompts:
                    data = {
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    response = requests.post(url, headers=headers, json=data).json()
                    responses.append(response["choices"][0]["message"]["content"])
            
            else:
                raise ValueError(f"Batch processing not implemented for model {model_name}")
            
            # Process all responses
            for question, hint_type, response in zip(batch_questions, batch_hint_types, responses):
                if question not in final_result:
                    final_result[question] = {}
                
                dict_value = {
                    'hint_response': response,
                    'real_answer': qa_dataset[question]
                }
                final_result[question][hint_type] = dict_value
                
                print(f"Question: {question[:100]}...")
                print(f"Response: {response}")
                print("-" * 50)
            
            req_success = True
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            print(f"Attempts remaining: {max_req_count}")
            if max_req_count > 0:
                max_req_count -= 1
            else:
                # Record errors for all failed questions
                for question, hint_type, prompt in zip(batch_questions, batch_hint_types, batch_prompts):
                    if question not in error_dataset:
                        error_dataset[question] = {}
                    error_dataset[question][hint_type] = {
                        'error_message': str(e),
                        'real_answer': qa_dataset[question],
                        'used_prompt': prompt
                    }
    
    return final_result, error_dataset

def calculate_result_per_question(
    model_name: str,
    question: str,
    prompt: str,
    final_result: Dict,
    error_dataset: Dict,
    qa_dataset: Dict,
    hint_type: str,
    task_type: str,
    use_cot: bool,
    openai_key: str,
    temperature: float = 0.0,
    model: Any = None,
    tokenizer: Any = None
) -> Tuple[Dict, Dict]:
    """
    Process a single question (wrapper around batch processing)
    """
    return process_batch(
        model_name=model_name,
        batch_questions=[question],
        batch_prompts=[prompt],
        batch_hint_types=[hint_type],
        final_result=final_result,
        error_dataset=error_dataset,
        qa_dataset=qa_dataset,
        task_type=task_type,
        use_cot=use_cot,
        openai_key=openai_key,
        temperature=temperature,
        model=model
    )
