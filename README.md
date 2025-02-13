# LLM Evaluation Framework

A framework for systematically evaluating Large Language Models across various benchmarks. Currently supports MMLU (Massive Multitask Language Understanding) evaluation with plans to expand to other benchmarks.

## Overview

This project provides a clean, modular framework for evaluating LLMs. Key features:
- Modular design separating evaluation logic from model handling
- Support for local and Hugging Face models
- Automated data management through Hugging Face datasets
- Detailed results logging and analysis
- Docker support for reproducible environments

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research.git
cd research
```

2. Set up the environment:

Using pip:
```bash
pip install -r requirements.txt
```

Using Docker:
```bash
docker compose up -d
```

3. Run an evaluation:
```bash
python scripts/qwen_mmlu_eval.py --model-path models/Qwen2.5-3B-Instruct
```

## Project Structure

```
research/
├── compose.yml           # Docker compose configuration
├── Dockerfile           # Docker build configuration
├── evaluations/         # Evaluation results directory
├── models/              # Local model storage
│   └── Qwen2.5-3B-Instruct/
├── src/
│   ├── eval/           # Evaluation implementations
│   └── scripts/        # Evaluation runner scripts
└── requirements.txt    # Python dependencies
```

## Documentation

Detailed documentation is available in the [Wiki](../../wiki).
