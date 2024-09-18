from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
# login()

model = "/home/lyb/workspace/llama3/llama3-8b"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
"text-generation",
      model=model,
      torch_dtype=torch.float16,
 device_map="auto",
)

text = "Provide the probability that the following answer for the following question is correct (0% to 100%). Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\n. Answer: 18"

sequences = pipeline(
    text,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=400,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")