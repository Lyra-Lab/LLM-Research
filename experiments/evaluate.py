# %%
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.notebook import tqdm

# %% Model loading
def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with multi-GPU support"""
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # Load model with optimal settings for T4 GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for T4 GPUs
        device_map="auto",         # Automatically handle multi-GPU
        max_memory={i: "12GiB" for i in range(n_gpus)},  # T4 has 16GB but leave some headroom
    )
    model.eval()

    # Get device (first GPU)
    device = torch.device("cuda:0")

    return model, tokenizer, device

# %% Data loading
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

# %% Evaluation
# Format prompt
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

# Main evaluation function
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

# 1. Load model
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Use the 1.5B variant
model, tokenizer, device = load_model_and_tokenizer(model_name)

# 2. Load data
eval_df_ru = load_mmlu_data(split="val", language='ru')

eval_df_ru.describe()

# 3. Run evaluation
results_df_ru = evaluate_model(model, tokenizer, eval_df_ru, device)

# 4. Calculate and display results for english
accuracy = results_df_ru['correct'].mean()
subject_accuracy = results_df_ru.groupby('subject')['correct'].mean()

# 5. Save Russian results
results_df_ru.to_csv(f"/content/mmlu_results_{model_name.replace('/', '_')}_ru.csv", index=False)

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

analyze_results("/content/mmlu_results_Qwen_Qwen2.5-1.5B-Instruct_ru.csv")

"""# LEP fine tuning for Russian"""

# TODO
