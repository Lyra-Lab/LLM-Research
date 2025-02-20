# %% Imports and Setup
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from typing import List, Optional
from tqdm import tqdm

# %% Corpus Loading Functions
def load_russian_corpus(
    dataset_name: str = "IlyaGusev/habr",
    text_field: str = "text_markdown",
    max_samples: Optional[int] = None
) -> List[str]:
    """Load a high-quality Russian text corpus."""
    # Load the dataset (using the "train" split)
    dataset = load_dataset(dataset_name, split="train")

    # Extract texts from the specified field
    texts = dataset[text_field]

    # If max_samples is set, take only that many samples
    if max_samples:
        texts = texts[:max_samples]

    return texts

# %% Tokenizer Training Functions
def train_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: str = "russian_qwen_tokenizer"
):
    """Train a new tokenizer based on Qwen2 architecture"""
    # Initialize a new tokenizer from local path
    new_tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen2.5-3B-Instruct",
        trust_remote_code=True
    )

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

# %% Tokenizer Testing Functions
def test_tokenizer(tokenizer, sample_texts: List[str]):
    """Test the trained tokenizer on sample texts"""
    results = []
    for text in sample_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        results.append({
            "original": text,
            "decoded": decoded,
            "num_tokens": len(tokens)
        })
        print(f"Original: {text}")
        print(f"Decoded: {decoded}")
        print(f"Number of tokens: {len(tokens)}\n")
    return results

# %% Main Execution
# 1. Load corpus
print("Loading Russian corpus...")
texts = load_russian_corpus(max_samples=100000)

# 2. Train tokenizer
print("Training tokenizer...")
tokenizer = train_tokenizer(texts)

# 3. Test tokenizer
print("Testing tokenizer...")
sample_texts = [
    "Привет, как дела?",
    "Машинное обучение - это интересно!",
    "Я изучаю обработку естественного языка."
]
test_results = test_tokenizer(tokenizer, sample_texts)
