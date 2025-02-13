import time
import torch
import numpy as np
import json
import os
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Loading the model from a local directory
# Load the Qwen/Qwen2.5-3B-Instruct model and tokenizer
work_dir = "/kaggle/working/"
# model_path = work_dir + "models/Qwen2.5-3B-Instruct"  # Path to your local model directory
model_path = work_dir + "models/Qwen2.5-1.5B-Instruct"
model_name = model_path.split('/')[-1]
eval_dir = work_dir + f"{model_name}-eval"
os.makedirs(eval_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # adjust based on your hardware
    device_map="auto"
)
model.eval()

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
mmlu_results = {'english': {}, 'russian': {}}

# English evaluation
for task in mmlu_tasks:
    try:
        dataset_mmlu = load_dataset("cais/mmlu", task, split="test")
        print(f"Evaluating on cais/mmlu dataset ({task})")
        accuracy_mmlu = evaluate_dataset(dataset_mmlu, model, tokenizer)
        print(f"Accuracy on cais/mmlu ({task}): {accuracy_mmlu:.2f}%")
        mmlu_results['english'][task] = accuracy_mmlu
    except Exception as e:
        print(f"Error evaluating {task}: {str(e)}")

mmlu_results['english']["average"] = sum(mmlu_results['english'].values()) / len(mmlu_results['english'])

# Russian evaluation
for task in mmlu_tasks:
    try:
        dataset_mmlu = load_dataset("cais/mmlu", task, split="test")
        print(f"Evaluating on cais/mmlu dataset ({task})")
        accuracy_mmlu = evaluate_dataset(dataset_mmlu, model, tokenizer)
        print(f"Accuracy on cais/mmlu ({task}): {accuracy_mmlu:.2f}%")
        mmlu_results['russian'][task] = accuracy_mmlu
    except Exception as e:
        print(f"Error evaluating {task}: {str(e)}")

mmlu_results['russian']["average"] = sum(mmlu_results['russian'].values()) / len(mmlu_results['russian'])

# Save MMLU results
with open(f"{eval_dir}/mmlu.json", 'w') as f:
    json.dump(mmlu_results, f, indent=2)

def measure_generation_performance(
    model,
    tokenizer,
    text: str,
    n_runs: int = 5
) -> Dict[str, float]:
    """Measure generation performance metrics for a given text."""
    metrics = {
        'token_count': [],
        'generation_time': [],
        'tokens_per_second': []
    }

    # Tokenize once to get input token count
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_token_count = inputs.input_ids.size(1)

    # Run multiple times to get stable measurements
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Adjust based on your needs
                temperature=0.7
            )
        generation_time = time.time() - start_time

        output_token_count = outputs.size(1) - input_token_count
        tokens_per_second = output_token_count / generation_time if generation_time > 0 else 0

        metrics['token_count'].append(output_token_count)
        metrics['generation_time'].append(generation_time)
        metrics['tokens_per_second'].append(tokens_per_second)

    return {
        'input_tokens': input_token_count,
        'avg_output_tokens': np.mean(metrics['token_count']),
        'avg_generation_time': np.mean(metrics['generation_time']),
        'avg_tokens_per_second': np.mean(metrics['tokens_per_second']),
        'std_tokens_per_second': np.std(metrics['tokens_per_second'])
    }

def compare_language_performance(model, tokenizer):
    """Compare performance between English and Russian text generation."""

    # Sample texts (roughly equivalent content)
    texts = {
        'english': """
        Please write a short story about a cat who discovers a magical garden.
        The story should be appropriate for children and include some description
        of the flowers and plants the cat encounters.
        """,

        'russian': """
        Пожалуйста, напишите короткий рассказ о коте, который обнаруживает волшебный сад.
        Рассказ должен быть подходящим для детей и включать описание
        цветов и растений, которые встречает кот.
        """
    }

    # Measure performance for each language
    speed_results = {}
    for lang, text in texts.items():
        print(f"\nMeasuring {lang} performance...")
        speed_results[lang] = measure_generation_performance(model, tokenizer, text)

    # Print comparative results
    print("\n=== Performance Comparison ===")
    metrics = ['input_tokens', 'avg_output_tokens', 'avg_generation_time',
               'avg_tokens_per_second']

    for metric in metrics:
        print(f"\n{metric}:")
        for lang in speed_results:
            print(f"{lang}: {speed_results[lang][metric]:.2f}")

        # Calculate relative difference
        if 'english' in speed_results and 'russian' in speed_results:
            diff_percent = ((speed_results['russian'][metric] - speed_results['english'][metric])
                          / speed_results['english'][metric] * 100)
            print(f"Relative difference: {diff_percent:+.1f}%")

    # Save speed results
    with open(f"{eval_dir}/speed.json", 'w') as f:
        json.dump(speed_results, f, indent=2)

    return speed_results

results = compare_language_performance(model, tokenizer)

# Load the data
mmlu_data = mmlu_results  # Use results from previous evaluation
speed_data = results  # Use results from previous comparison

# Set a modern color palette
colors = {
    'english': '#00A67E',  # Teal
    'russian': '#FF6B6B',  # Coral
    'background': '#FFFFFF',
    'grid': '#E5E5E5',
    'text': '#2D3436'
}

# Custom template for consistent styling
template = go.layout.Template()
template.layout.plot_bgcolor = colors['background']
template.layout.paper_bgcolor = colors['background']
template.layout.font = dict(color=colors['text'], family="Inter, sans-serif")

def create_performance_comparison():
    """Creates a bar chart comparing performance metrics between languages"""
    metrics = pd.DataFrame(speed_data).round(2)
    metrics = metrics.transpose()

    fig = go.Figure()

    # Add bars for each metric
    for column in metrics.columns:
        fig.add_trace(go.Bar(
            name=column.replace('_', ' ').title(),
            x=metrics.index,
            y=metrics[column],
            text=metrics[column],
            textposition='auto',
        ))

    fig.update_layout(
        title="Language Performance Comparison",
        template=template,
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.write_html(os.path.join(eval_dir, "performance_comparison.html"))

def create_mmlu_comparison():
    """Creates a scatter plot comparing MMLU scores between languages"""
    # Convert to DataFrame
    df = pd.DataFrame({
        'task': list(mmlu_data['english'].keys())[:-1],  # Exclude average
        'english': list(mmlu_data['english'].values())[:-1],
        'russian': list(mmlu_data['russian'].values())[:-1]
    })

    # Sort by English score
    df = df.sort_values('english', ascending=True)

    fig = go.Figure()

    # Add lines and markers
    for lang in ['english', 'russian']:
        fig.add_trace(go.Scatter(
            x=df[lang],
            y=df['task'],
            name=lang.capitalize(),
            mode='markers',
            marker=dict(
                size=8,
                color=colors[lang],
            )
        ))

    fig.update_layout(
        title="MMLU Score Comparison by Task",
        xaxis_title="Score (%)",
        yaxis_title="Task",
        template=template,
        height=1500,  # Taller to accommodate all tasks
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.write_html(os.path.join(eval_dir, "mmlu_comparison.html"))

def create_category_performance():
    """Creates a radar chart for main category performance"""
    # Group tasks into categories
    categories = {
        'Mathematics': ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics'],
        'Sciences': ['astronomy', 'college_physics', 'high_school_biology', 'high_school_chemistry', 'high_school_physics'],
        'Computer Science': ['college_computer_science', 'computer_security', 'machine_learning'],
        'Medicine': ['anatomy', 'clinical_knowledge', 'college_medicine', 'medical_genetics'],
        'Humanities': ['philosophy', 'world_religions', 'high_school_european_history', 'high_school_world_history'],
        'Social Sciences': ['sociology', 'psychology', 'economics', 'political_science']
    }

    # Calculate average scores for each category
    category_scores = {lang: {} for lang in ['english', 'russian']}
    for lang in ['english', 'russian']:
        for category, tasks in categories.items():
            scores = [mmlu_data[lang][task] for task in tasks if task in mmlu_data[lang]]
            category_scores[lang][category] = np.mean(scores)

    # Create radar chart
    fig = go.Figure()

    for lang in ['english', 'russian']:
        fig.add_trace(go.Scatterpolar(
            r=list(category_scores[lang].values()),
            theta=list(category_scores[lang].keys()),
            name=lang.capitalize(),
            fill='toself',
            line_color=colors[lang]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Performance by Category",
        template=template,
        height=600
    )

    fig.write_html(os.path.join(eval_dir, "category_performance.html"))

# Generate all visualizations
create_performance_comparison()
create_mmlu_comparison()
create_category_performance()
