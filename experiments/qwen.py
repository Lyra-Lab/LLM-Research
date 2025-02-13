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

# %% [code]
def construct_prompt(example):
    """
    Constructs a prompt from a dataset example.
    Handles cases where 'choices' key is not present, using 'options' instead.
    If neither is present, tries to use 'possible_answers'.
    **Also handles the case where the key is 'answers' instead of 'choices', 'options', or 'possible_answers'.**
    Assumes the following keys in example:
      - "question": the question text.
      - "choices" or "options" or "possible_answers" **or "answers"**: a list of answer choices.
      - "answer": the correct answer letter (e.g., "A").
    """
    # Check for all possible keys
    choices_key = None
    for key in ["choices", "options", "possible_answers", "answers"]:
        if key in example:
            choices_key = key
            break

    # Raise KeyError if none of the expected keys are found
    if choices_key is None:
        raise KeyError("Example must contain 'choices', 'options', 'possible_answers', or 'answers' key.")

    # Create the prompt with the question and enumerated choices.
    prompt = f"Question: {example['question']}\nChoices:\n"
    for idx, choice in enumerate(example[choices_key]):
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
        # Convert the ground truth answer to a letter if it's an integer.
        ground_truth_answer = example["answer"]
        if isinstance(ground_truth_answer, int):
            ground_truth_answer = chr(65 + ground_truth_answer)  # Convert 0 to A, 1 to B, etc.
        else:

            ground_truth_answer = ground_truth_answer.strip().upper()

        if answer_pred == ground_truth_answer:
            correct += 1
    accuracy = correct / total * 100
    return accuracy

# %% [code]
# English Evaluation (cais/mmlu)
# List of all MMLU tasks
mmlu_tasks = [
    'abstract_algebra', 'anatomy', 'astronomy', 'auxiliary_train',
    'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
    'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering',
    'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history',
    'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics',
    'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

# Evaluate the model on all MMLU tasks
results = {}
for task in mmlu_tasks:
    try:
        dataset_mmlu = load_dataset("cais/mmlu", task, split="test")
        print(f"Evaluating on cais/mmlu dataset ({task})")
        accuracy_mmlu = evaluate_dataset(dataset_mmlu, model, tokenizer)
        print(f"Accuracy on cais/mmlu ({task}): {accuracy_mmlu:.2f}%")
        results[task] = accuracy_mmlu
    except Exception as e:
        print(f"Error evaluating {task}: {str(e)}")


# Print the results
print("\nResults:")
for task, accuracy in results.items():
    print(f"{task}: {accuracy:.2f}%")


results["average"] = sum(results.values()) / len(results)
print(f"Average accuracy: {results['average']}")

# %% [code]
# This is the same but with the mmlu_ru dataset (not tested currently)
# Russian Evaluation (mmlu_ru)
# List of all MMLU tasks
mmlu_ru_tasks = [
    'abstract_algebra' 'anatomy', 'astronomy', 'auxiliary_train',
    'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics',
    'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering',
    'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history',
    'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics',
    'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
    'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
    'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
    'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

# Evaluate the model on all MMLU tasks
results = {}
for task in mmlu_ru_tasks:
    try:
        dataset_mmlu = load_dataset("cais/mmlu", task, split="test")
        print(f"Evaluating on cais/mmlu dataset ({task})")
        accuracy_mmlu = evaluate_dataset(dataset_mmlu, model, tokenizer)
        print(f"Accuracy on cais/mmlu ({task}): {accuracy_mmlu:.2f}%")
        results[task] = accuracy_mmlu
    except Exception as e:
        print(f"Error evaluating {task}: {str(e)}")


# Print the results
print("\nResults:")
for task, accuracy in results.items():
    print(f"{task}: {accuracy:.2f}%")


results["average"] = sum(results.values()) / len(results)
print(f"Average accuracy: {results['average']}")
