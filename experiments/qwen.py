# %% [code]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Load the Qwen/Qwen2.5-3B-Instruct model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # adjust based on your hardware
    device_map="auto"
)
model.eval()

def construct_prompt(example):
    """
    Constructs a prompt from a dataset example.
    Assumes the following keys in example:
      - "question": the question text.
      - "choices": a list of answer choices.
      - "answer": the correct answer letter (e.g., "A").
    """
    # Create the prompt with the question and enumerated choices.
    prompt = f"Question: {example['question']}\nChoices:\n"
    for idx, choice in enumerate(example['choices']):
        # Convert index 0,1,2... to A, B, C, etc.
        letter = chr(65 + idx)
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"  # model completion should output the answer letter.
    return prompt

def evaluate_dataset(dataset, model, tokenizer):
    """
    Iterates over the dataset, generates the model output for each prompt,
    extracts the predicted answer (first capital letter found), and compares
    it to the ground truth to compute accuracy.
    """
    correct = 0
    total = len(dataset)
    for example in tqdm(dataset, desc="Evaluating"):
        prompt = construct_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Generate a short output; use deterministic generation for evaluation.
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False
        )
        # Decode the generated tokens.
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the text generated after the prompt.
        generated_text = response[len(prompt):].strip()
        # Find the first capital letter (A, B, C, …) as predicted answer.
        answer_pred = ""
        for char in generated_text:
            if char.upper() in ["A", "B", "C", "D", "E", "F", "G"]:
                answer_pred = char.upper()
                break
        # Compare with the ground truth answer.
        if answer_pred == example["answer"].strip().upper():
            correct += 1
    accuracy = correct / total * 100
    return accuracy

# %% [code]
# Download and load datasets from Hugging Face
# Adjust the split as necessary – here we use "test" split assuming evaluation examples are stored there.
dataset_mmlu = load_dataset("cais/mmlu", split="test")
dataset_mmlupro = load_dataset("Vikhrmodels/mmlupro-ru", split="test")

print("Evaluating on cais/mmlu dataset")
accuracy_mmlu = evaluate_dataset(dataset_mmlu, model, tokenizer)
print(f"Accuracy on cais/mmlu: {accuracy_mmlu:.2f}%")

print("\nEvaluating on Vikhrmodels/mmlupro-ru dataset")
accuracy_mmlupro = evaluate_dataset(dataset_mmlupro, model, tokenizer)
print(f"Accuracy on Vikhrmodels/mmlupro-ru: {accuracy_mmlupro:.2f}%")
