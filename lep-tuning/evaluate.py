# %% Imports and Setup
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast,
from tqdm import tqdm
from typing import List, Optional
from pathlib import Path
import tarfile
import zipfile

# %% Model Loading Functions
def load_model_and_tokenizer(model_path="models/Qwen2.5-1.5B-Instruct"):
    """Load model and tokenizer with multi-GPU support"""
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")

    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True  # Important for Qwen models
    )

    # Load model with optimal settings for T4 GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for T4 GPUs
        device_map="auto",         # Automatically handle multi-GPU
        max_memory={i: "12GiB" for i in range(n_gpus)},  # T4 has 16GB but leave some headroom
        trust_remote_code=True  # Important for Qwen models
    )
    model.eval()

    # Get device (first GPU)
    device = torch.device("cuda:0")

    return model, tokenizer, device

# %% Data Loading Functions
def load_mmlu_data(subjects=None, split="val", language="ru"):
    """Load MMLU_RU data for specified subjects and language."""
    if subjects is None:
        subjects = [
            'abstract_algebra',
            'college_computer_science',
            'college_mathematics',
            'formal_logic',
            'machine_learning',
            'college_physics',
            'high_school_statistics',
            'electrical_engineering',
            'computer_security'
        ]

    dfs = []
    for subject in subjects:
        try:
            dataset = load_dataset("NLPCoreTeam/mmlu_ru", subject, split=split)
            df = dataset.to_pandas()

            # Map integer answers to corresponding string labels
            int2str = dataset.features['answer'].int2str
            df['answer'] = df['answer'].map(int2str)

            # Insert subject column
            df.insert(0, 'subject', subject)

            # Keep only the selected language's question and choices
            lang_suffix = "_ru" if language == "ru" else "_en"
            df = df.rename(columns={
                f'question{lang_suffix}': 'question',
                f'choices{lang_suffix}': 'choices'
            })[['subject', 'question', 'choices', 'answer']]

            dfs.append(df)
        except Exception as e:
            print(f"Error loading {subject} ({language}): {e}")

    return pd.concat(dfs, ignore_index=True)

# %%  Prompt Templates and Formatting
PROMPTS = {
    "ru": {
        "template": (
            "Ответьте на вопрос, выбрав правильный вариант (A, B, C или D).\n"
            "Вопрос: {question}\n"
            "Варианты ответа:\n"
            "{options}\n"
            "Ответ (укажите только букву A, B, C или D):"
        ),
        "question_key": "question",
        "choices_key": "choices"
    },
    "en": {
        "template": (
            "Answer the question by selecting the correct option (A, B, C, or D).\n"
            "Question: {question}\n"
            "Options:\n"
            "{options}\n"
            "Answer (provide only the letter A, B, C, or D):"
        ),
        "question_key": "question",
        "choices_key": "choices"
    }
}

def format_prompt(row):
    """Format a prompt for a question row using its language settings."""
    lang = row.get("language", "ru")
    config = PROMPTS.get(lang, PROMPTS["ru"])
    question = row[config["question_key"]]
    choices = row[config["choices_key"]]
    options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    return config["template"].format(question=question, options=options)

# %%  Model Evaluation
def evaluate_model(model, tokenizer, df, device, debug_samples=-1, batch_size=4):
    """Evaluate model on the dataset with debugging information and batch processing."""
    results = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i : i + batch_size]
        # Generate prompts for the batch
        prompts = [format_prompt(row) for _, row in batch_df.iterrows()]

        # Tokenize batch and move to device
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            except Exception as e:
                print("Error during model generation:", e)
                continue

        # Process each sample in the batch
        for idx, (prompt, (_, row)) in enumerate(zip(prompts, batch_df.iterrows())):
            response = tokenizer.decode(outputs[idx], skip_special_tokens=True)
            # Remove prompt text from response to get only the generated part
            generated = response[len(prompt):].strip()

            # Get and process the correct answer
            correct_answer = row["answer"]
            if str(correct_answer).isdigit():
                correct_answer = chr(65 + int(correct_answer))
            correct_answer = str(correct_answer).upper()

            # Extract the first valid answer (A, B, C, or D) from the generated text
            pred = next((c for c in generated if c.upper() in "ABCD"), "X").upper()

            # Use default keys, assuming Russian data if specific keys are missing
            question_text = row.get("question_ru", row.get("question"))
            choices = row.get("choices_ru", row.get("choices"))
            language = row.get("language", "ru")

            if idx < debug_samples:
                print(f"\nDebug Sample {idx + 1} ({language}):")
                print(f"Question: {question_text}")
                print(f"Full Response: {generated}")
                print(f"Extracted Prediction: {pred}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Choices: {choices}")

            results.append({
                "subject": row["subject"],
                "language": language,
                "question": question_text,
                "correct_answer": correct_answer,
                "predicted_answer": pred,
                "full_response": generated,
                "correct": pred == correct_answer,
            })

    return pd.DataFrame(results)

# %%  Results Analysis
def analyze_results(csv_file):
    """Analyze the MMLU results CSV file."""
    # Load the data
    df = pd.read_csv(csv_file)

    # Ensure correct column exists
    if 'correct' not in df.columns:
        raise ValueError("CSV file must contain a 'correct' column with boolean values.")

    # Overall accuracy
    overall_accuracy = df['correct'].mean() * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Accuracy by subject
    subject_accuracy = df.groupby('subject')['correct'].mean() * 100
    print("\nAccuracy by Subject:")
    print(subject_accuracy)

    # Most and least accurate subjects
    most_accurate = subject_accuracy.idxmax()
    least_accurate = subject_accuracy.idxmin()

    print(f"\nBest Performing Subject: {most_accurate} ({subject_accuracy.max():.2f}%)")
    print(f"Worst Performing Subject: {least_accurate} ({subject_accuracy.min():.2f}%)")

    # Common mistakes analysis
    incorrect_df = df[df['correct'] == False]
    if not incorrect_df.empty:
        print("\nSample Incorrect Predictions:")
        print(incorrect_df[['subject', 'question', 'correct_answer', 'predicted_answer']].sample(min(5, len(incorrect_df))))
    else:
        print("No incorrect predictions found!")

    return subject_accuracy

# %%  Tokenizer Training
def load_russian_corpus(dataset_name: str = "IlyaGusev/habr",
                       text_field: str = "text_markdown",
                       max_samples: Optional[int] = None) -> List[str]:
    """Load a high-quality Russian text corpus."""
    # Load the dataset (using the "train" split)
    dataset = load_dataset(dataset_name, split="train")

    # Extract texts from the specified field
    texts = dataset[text_field]

    # If max_samples is set, take only that many samples
    if max_samples:
        texts = texts[:max_samples]

    return texts

def train_tokenizer(texts: List[str],
                   vocab_size: int = 32000,
                   min_frequency: int = 2,
                   output_dir: str = "russian_qwen_tokenizer"):
    """Train a new tokenizer based on Qwen2 architecture"""
    # Initialize a new tokenizer
    new_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    # Train tokenizer on the Russian corpus
    new_tokenizer.train_new_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True
    )

    # Save the tokenizer
    os.makedirs(output_dir, exist_ok=True)
    new_tokenizer.save_pretrained(output_dir)

    return new_tokenizer

def test_tokenizer(tokenizer, sample_texts: List[str]):
    """Test the trained tokenizer on sample texts"""
    for text in sample_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"Original: {text}")
        print(f"Decoded: {decoded}")
        print(f"Number of tokens: {len(tokens)}\n")

# Main Execution
# %% 1. Load model
model_path = "models/Qwen2.5-1.5B-Instruct"  # Relative path from your working directory
model, tokenizer, device = load_model_and_tokenizer(model_path)

# %% 2. Load data
eval_df_ru = load_mmlu_data(split="val", language='ru')

# %% 3. Run evaluation
results_df_ru = evaluate_model(model, tokenizer, eval_df_ru, device)

# %% 4. Save and analyze results
results_df_ru.to_csv(f"/content/mmlu_results_{model_name.replace('/', '_')}_ru.csv",
                    index=False)
analyze_results(f"/content/mmlu_results_{model_name.replace('/', '_')}_ru.csv")

# %% 5. Train and test tokenizer
texts = load_russian_corpus(max_samples=100000)
tokenizer = train_tokenizer(texts)
sample_texts = [
    "Привет, как дела?",
    "Машинное обучение - это интересно!",
    "Я изучаю обработку естественного языка."
]
test_tokenizer(tokenizer, sample_texts)
