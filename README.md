
# ASLLFT: Agentic System for Domain-Aware Fine-Tuning of LLMs

ASLLFT (Automated System for LLM Fine-Tuning) is an end-to-end agentic framework designed to automate the fine-tuning of large language models (LLMs) for domain-specific tasks. It intelligently selects training configurations, evaluates model performance across multiple criteria, and adaptively refines the process with minimal human intervention.

## Key Features

- **Agentic Coordination**: A planning LLM manages the entire pipeline—domain detection, configuration, training, evaluation, and retraining.
- **Domain-Aware Fine-Tuning**: Automatically detects the dataset’s domain and injects role-specific context into training and evaluation prompts.
- **Parameter-Efficient Training (PEFT)**: Uses LoRA via Unsloth and 4-bit quantization for efficient fine-tuning on limited hardware.
- **Closed-Loop Optimization**: Incorporates multi-dimensional evaluation and feedback to guide adaptive retries.
- **Resource-Efficient**: Capable of running on consumer-grade GPUs (e.g., Google Colab T4).

## System Overview

### Components

- **Domain Injection Agent**: Detects domain and sets expert persona for training prompts.
- **Fine-Tuning Agent**: Suggests initial hyperparameters based on dataset characteristics and system constraints.
- **Training Module**: Performs PEFT-based fine-tuning using Unsloth, Transformers, and TRL.
- **Evaluation Agent**: Uses an LLM as an expert judge to score model responses.
- **Adaptive Retry Planner**: Analyzes evaluation feedback and training logs to recommend improved configurations.
- **Memory Module**: Tracks strategy, training metrics, evaluation scores, and qualitative feedback to guide decisions across retries.

## Results Summary

ASLLFT demonstrates progressive improvement over retries, effective domain-specific customization, and robust agent coordination.

| Metric        | Attempt 1 | Attempt 2 |
|---------------|-----------|-----------|
| Accuracy      | 6.80      | 7.75      |
| Reasoning     | 6.00      | 6.75      |
| Completeness  | 8.60      | 8.75      |
| Clarity       | 7.40      | 8.00      |
| **Average**   | **7.20**  | **7.81**  |

## Setup and Usage

### Requirements

- Python ≥ 3.9
- Google Colab with T4 GPU (or equivalent)
- Required packages: `transformers`, `datasets`, `peft`, `trl`, `unsloth`, `google-generativeai`

### Installation

```bash
pip install transformers datasets peft trl unsloth google-generativeai
```

### Running ASLLFT

1. Open `ASLLFT.ipynb` in a Colab or Jupyter environment.
2. Configure API access (for Gemini, if used).
3. Upload a training dataset.
4. Execute cells to run the pipeline:
   - Domain detection
   - Hyperparameter suggestion
   - Fine-tuning
   - Multi-dimensional evaluation
   - Retry with improved configuration (up to 2 additional times)

## Supported Datasets

Example datasets tested with ASLLFT:
- `FreedomIntelligence/medical-o1-reasoning-SFT`
- `chemouda/legal_reason`

The system automatically identifies the domain (e.g., medical, legal) and injects relevant context into prompts and evaluation.

ASLLFT ensures that domain-specific context is injected automatically, while maintaining a generalized and reproducible pipeline across tasks.

## Future Work

- Replace Gemini API with a self-hosted LLM for evaluation and planning.
- Extend support to additional PEFT techniques and full fine-tuning.
- Incorporate advanced HPO methods (e.g., Bayesian Optimization).
- Improve evaluation reliability with multi-judge and hallucination checks.
- Explore RLAIF (Reinforcement Learning from AI Feedback) integration.
