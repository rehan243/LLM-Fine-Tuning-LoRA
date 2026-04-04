# LLM-Fine-Tuning-LoRA

Fine-tuning Large Language Models (LLaMA, Mistral) with parameter-efficient methods (LoRA, QLoRA, PEFT) for domain-specific enterprise use cases.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

---

## Overview

Production-ready pipeline for fine-tuning open-source LLMs using parameter-efficient techniques. Achieves domain-specific performance comparable to hosted APIs while reducing inference costs by 40% and maintaining full data privacy.

Developed and deployed at **Verticiti** and **Reallytics.ai** for enterprise clients requiring domain-specific language models in regulated industries.

## Supported Methods

| Method | Description | Memory Reduction | Use Case |
|---|---|---|---|
| **LoRA** | Low-Rank Adaptation of attention matrices | ~60% | General fine-tuning |
| **QLoRA** | Quantized LoRA with 4-bit base model | ~75% | Memory-constrained environments |
| **PEFT** | Parameter-Efficient Fine-Tuning framework | ~65% | Multi-task adaptation |

## Architecture

```
┌──────────────────────────────────────────┐
│            Training Pipeline              │
│                                          │
│  ┌──────────┐    ┌───────────────────┐   │
│  │ Dataset  │───▶│  Tokenization &   │   │
│  │ Loader   │    │  Preprocessing    │   │
│  └──────────┘    └────────┬──────────┘   │
│                           │              │
│  ┌────────────────────────▼──────────┐   │
│  │       Base Model Loading          │   │
│  │  (LLaMA / Mistral / Falcon)       │   │
│  │  4-bit quantized (bitsandbytes)   │   │
│  └────────────────────────┬──────────┘   │
│                           │              │
│  ┌────────────────────────▼──────────┐   │
│  │     LoRA Adapter Injection        │   │
│  │  - Target: q_proj, v_proj, k_proj │   │
│  │  - Rank: 16-64                    │   │
│  │  - Alpha: 32-128                  │   │
│  └────────────────────────┬──────────┘   │
│                           │              │
│  ┌────────────────────────▼──────────┐   │
│  │      SFTTrainer (HuggingFace)     │   │
│  │  - Gradient accumulation          │   │
│  │  - Mixed precision (bf16)         │   │
│  │  - Cosine LR scheduler           │   │
│  └────────────────────────┬──────────┘   │
│                           │              │
│  ┌────────────────────────▼──────────┐   │
│  │     Evaluation & Metrics          │   │
│  │  - Perplexity, BLEU, ROUGE       │   │
│  │  - Domain-specific benchmarks     │   │
│  └───────────────────────────────────┘   │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│           Serving Pipeline                │
│                                          │
│  ┌───────────┐   ┌──────────────────┐    │
│  │  VLLM     │──▶│  FastAPI Server  │    │
│  │  Engine   │   │  (REST + gRPC)   │    │
│  └───────────┘   └──────────────────┘    │
│                                          │
│  Deployed on: AWS SageMaker / ECS+Docker │
└──────────────────────────────────────────┘
```

## Key Features

- **Multi-Model Support**: Fine-tune LLaMA-2 (7B/13B/70B), Mistral-7B, Falcon, and other HuggingFace models
- **Memory Efficient**: QLoRA enables fine-tuning 70B models on a single A100 GPU
- **Production Serving**: VLLM integration for optimized GPU utilization and high-throughput inference
- **CUDA Optimized**: Custom CUDA kernels for attention computation and quantization
- **Automated Pipeline**: End-to-end from data preparation to model deployment
- **Evaluation Suite**: Comprehensive benchmarking with perplexity, BLEU, ROUGE, and custom domain metrics
- **Cost Reduction**: 40% reduction vs hosted API costs (GPT-4, Claude)
- **Docker + SageMaker**: Containerized deployment on AWS ECS/ECR or SageMaker endpoints

## Tech Stack

| Category | Technologies |
|---|---|
| **Framework** | HuggingFace Transformers, PEFT, TRL |
| **Models** | LLaMA-2, Mistral-7B, Falcon |
| **Quantization** | bitsandbytes (4-bit, 8-bit), GPTQ |
| **Serving** | VLLM, Text Generation Inference |
| **Compute** | CUDA, PyTorch, Mixed Precision (bf16/fp16) |
| **Cloud** | AWS SageMaker, ECS/ECR, Docker |
| **Monitoring** | Weights & Biases, TensorBoard |

## Project Structure

```
llm-fine-tuning-lora/
├── configs/
│   ├── lora_config.yaml
│   ├── qlora_config.yaml
│   └── training_args.yaml
├── data/
│   ├── prepare_dataset.py
│   ├── tokenize.py
│   └── data_quality.py
├── training/
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── trainer_utils.py
│   └── callbacks.py
├── evaluation/
│   ├── benchmark.py
│   ├── perplexity.py
│   └── domain_eval.py
├── serving/
│   ├── vllm_server.py
│   ├── fastapi_app.py
│   └── sagemaker_deploy.py
├── infrastructure/
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_evaluation.ipynb
├── tests/
├── requirements.txt
└── README.md
```

## Results

| Model | Method | Training Time | GPU Memory | Cost vs GPT-4 API |
|---|---|---|---|---|
| LLaMA-2 7B | LoRA | 4 hours | 16GB | -60% |
| LLaMA-2 13B | QLoRA | 8 hours | 24GB | -50% |
| Mistral 7B | LoRA | 3.5 hours | 14GB | -65% |
| LLaMA-2 70B | QLoRA | 24 hours | 48GB | -40% |

## Quick Start

```bash
git clone https://github.com/rehan243/LLM-Fine-Tuning-LoRA.git
cd LLM-Fine-Tuning-LoRA

pip install -r requirements.txt

# Fine-tune with LoRA
python training/train_lora.py --config configs/lora_config.yaml

# Fine-tune with QLoRA (memory efficient)
python training/train_qlora.py --config configs/qlora_config.yaml

# Serve with VLLM
python serving/vllm_server.py --model ./output/merged_model
```

## Author

**Rehan Malik** - CTO @ Reallytics.ai

- [LinkedIn](https://linkedin.com/in/rehan-malik-cto)
- [GitHub](https://github.com/rehan243)
- [Email](mailto:rehanmalil99@gmail.com)

---

