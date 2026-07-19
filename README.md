# Fine-Tuned Gemma-3 with RAG Integration

This repository contains the fine-tuning and retrieval-augmented generation (RAG) pipeline I built during my summer internship at İletişim Yazılım (2025). The goal was to adapt the Gemma-3-4B model to a specific set of technical topics (LLMs, machine learning, computer vision, data mining, databases) and combine it with a document retrieval system so the model can ground its answers in an external knowledge base instead of relying only on what it learned during training.

## Project Overview

- Base model: `google/gemma-3-4b-it` (loaded through `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`, 4-bit quantized)
- Fine-tuning method: LoRA, via the Unsloth library
- Vector database: ChromaDB
- Embedding model: `intfloat/multilingual-e5-small`
- Interface: Gradio

The original instruction-response dataset had 199 examples. It was expanded through rephrasing-based augmentation before training, validation and testing.

## Repository Contents

| File | Description |
|---|---|
| `newdset-early-val-aug-var-llmstaj.ipynb` | Main notebook: data loading, augmentation, LoRA fine-tuning, RAG pipeline, evaluation |
| `dataset.jsonl` | Original instruction-response dataset used for fine-tuning |
| `knowledge_base.json` | Knowledge base documents used by the RAG system, grouped by topic |
| `final_evaluation_report.json` | Training and RAG evaluation metrics from the final run |
| `sample_rag_outputs.json` | Example query/response/context outputs from the RAG pipeline |
| `training_metrics.png` | Training and validation loss curves |
| `rag_performance_analysis.png` | RAG evaluation metrics (semantic similarity, retrieval quality, context utilization, etc.) |
| `__huggingface_repos__.json` | Hugging Face model/repo identifiers used in the project |

## Setup

1. Clone the repository.
2. Install the dependencies (Unsloth, transformers, trl, chromadb, sentence-transformers, torch, pandas, scikit-learn).
3. Copy `llm_token_example.txt`, rename it, and put your own Hugging Face access token inside. Do not commit your real token.
4. Open the notebook and update the input paths (`dataset.jsonl`, `knowledge_base.json`) to match your local or Kaggle/Colab environment.
5. Run the notebook cells in order: setup, data loading and augmentation, fine-tuning, RAG setup, evaluation.

## Notes

- The notebook was originally run on Kaggle, so some file paths reference `/kaggle/input/...`. Update these paths if running locally or on another platform.
- GPU is required for fine-tuning; the notebook was tested with a single GPU using 4-bit quantization.
- Note: Model checkpoints, adapter weights, and compilation cache are intentionally excluded from this repository due to GitHub file size limits. This repository focuses on the implementation, evaluation, and reproducibility of the RAG pipeline.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
