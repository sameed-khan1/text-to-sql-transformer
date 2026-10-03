# 🔍 Text-to-SQL Transformer from Scratch

> Converting natural language questions into SQL queries using the WikiSQL dataset — built entirely from scratch with PyTorch, no pretrained weights, no `nn.Transformer`.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)
[![WikiSQL](https://img.shields.io/badge/Dataset-WikiSQL-green)](https://github.com/salesforce/WikiSQL)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)](https://streamlit.io)

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Environment Setup](#environment-setup)
5. [Data Setup](#data-setup)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Correctness Checks](#correctness-checks)
9. [Failure Cases](#failure-cases)

---

## Overview

Most of the world's useful data sits in database tables, and most people who need it cannot write SQL. This project builds an encoder–decoder Transformer that turns an English question about a table into the SQL query that answers it.

**Input:** An English question + column names of a table  
**Output:** A SQL query over that table

### Example
| | |
|---|---|
| **Table columns** | Player, No., Nationality, Position, Years in Toronto, School/Club Team |
| **Question** | What is Terrence Ross' nationality? |
| **Generated SQL** | `SELECT Nationality FROM table WHERE Player = 'terrence ross'` |

### Column-Token Framing
Columns are referred to by special tokens `<c0>`, `<c1>`, … so the model never has to spell a column name — it only needs to point at the right column.

```
Source: what is terrence ross' nationality <sep> <c0> player <c1> no. <c2> nationality ...
Target: select <c2> where <c0> = terrence ross
```

---

## Architecture

Built from scratch following **Vaswani et al. (2017) — Attention Is All You Need**.

```
Input Question + Columns
         │
    [InputLayer]          ← Token Embedding × √d_model + Positional Encoding
         │
  ┌──────────────┐
  │   ENCODER    │  × 3 layers
  │              │
  │ Self-Attention│
  │  Add & Norm  │
  │  Feed-Forward│
  │  Add & Norm  │
  └──────┬───────┘
         │ memory
  ┌──────▼───────┐
  │   DECODER    │  × 3 layers
  │              │
  │Masked Self-Attn│
  │  Add & Norm  │
  │ Cross-Attention│ ← attends to encoder memory
  │  Add & Norm  │
  │  Feed-Forward│
  │  Add & Norm  │
  └──────┬───────┘
         │
  [Linear + Softmax]     ← weight-tied to embedding
         │
     SQL Tokens
```

### Configuration

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Attention heads | 4 (d_k = d_v = 64) |
| Encoder layers | 3 |
| Decoder layers | 3 |
| Feed-forward size | 1024 |
| Dropout | 0.1 |
| Normalisation | Post-norm |
| Weight sharing | Encoder emb = Decoder emb = Output projection |
| **Trainable parameters** | **7,577,600** |

---

## Repository Layout

```
starter/
├── data_prep.py          # Load and preprocess WikiSQL data
├── tokenizer.py          # Train shared BPE vocabulary (8000 pieces)
├── dataset.py            # PyTorch Dataset + DataLoader
├── embeddings.py         # TokenEmbedding + PositionalEncoding + InputLayer
├── check_starter.py      # Verify starter code shapes
├── transformer.py        # Full model: attention, encoder, decoder, masks
├── train.py              # Training loop with Noam LR schedule
├── decode.py             # Greedy + beam search, SQL parser
├── evaluate_model.py     # Write prediction files, component accuracy, attention plot
├── app.py                # Streamlit web frontend
├── sql_converter.py      # Convert model output to readable SQL
├── best_model.pt         # Best saved checkpoint
├── sql_sp.model          # Trained SentencePiece tokenizer
├── results.json          # Training metrics and results
├── WikiSQL/              # WikiSQL data and official evaluator
│   └── data/
│       ├── train.jsonl / train.tables.jsonl / train.db
│       ├── dev.jsonl   / dev.tables.jsonl   / dev.db
│       └── test.jsonl  / test.tables.jsonl  / test.db
├── train_pairs.jsonl     # Preprocessed training pairs
├── dev_pairs.jsonl       # Preprocessed dev pairs
├── test_pairs.jsonl      # Preprocessed test pairs
├── dev_greedy.jsonl      # Greedy predictions (evaluator format)
├── dev_beam.jsonl        # Beam predictions (evaluator format)
├── test_beam.jsonl       # Test predictions (evaluator format)
└── results/
    ├── training_validation_loss.png
    ├── lr_schedule.png
    └── cross_attention.png
```

---

## Environment Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentencepiece records babel tqdm tabulate streamlit matplotlib
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Data Setup

```bash
git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
tar xvjf data.tar.bz2
cd ..
```

---

## How to Run

### Step 1 — Preprocess data
```bash
cd starter
python data_prep.py
python tokenizer.py
python check_starter.py
```

### Step 2 — Train
```bash
python train.py --scratch --epochs 30
```
Saves `best_model.pt`, `results.json`, `training_validation_loss.png`, `lr_schedule.png`.

> To resume training from a checkpoint:
> ```bash
> python train.py --resume --epochs 60
> ```

### Step 3 — Generate predictions
```bash
python evaluate_model.py
```
Writes `dev_greedy.jsonl`, `dev_beam.jsonl`, `test_beam.jsonl`, and `cross_attention.png`.

### Step 4 — Official evaluator
```bash
python WikiSQL/evaluate.py WikiSQL/data/dev.jsonl WikiSQL/data/dev.db dev_greedy.jsonl
python WikiSQL/evaluate.py WikiSQL/data/dev.jsonl WikiSQL/data/dev.db dev_beam.jsonl
python WikiSQL/evaluate.py WikiSQL/data/test.jsonl WikiSQL/data/test.db test_beam.jsonl
```

### Step 5 — Launch web app
```bash
streamlit run app.py
```

---

## Results

### Table 1 — Data

| Split | Pairs | Mean src len | Max src len | Mean tgt len | Max tgt len |
|---|---|---|---|---|---|
| Train | 56,336 | ~78 | 160 | ~10 | 64 |
| Dev | 8,421 | ~78 | 160 | ~10 | 64 |
| Test | 15,878 | ~78 | 160 | ~10 | 64 |

### Table 2 — Model and Training

| | |
|---|---|
| Trainable parameters | 7,577,600 |
| Epochs trained | 30 |
| Best epoch | 23 |
| Best dev loss | 2.0741 |
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9, Noam schedule, warmup=4000) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |

### Table 3 — Official Metrics

| Split | Decoding | Logical Form (%) | Execution (%) | Parse Failures (%) |
|---|---|---|---|---|
| Dev | Greedy | 10.08 | 16.55 | 0.00 |
| Dev | Beam (4) | 10.47 | 17.21 | 0.00 |
| Test | Beam (4) | 10.78 | 18.35 | 0.00 |

### Table 4 — Component Accuracy (Dev, Beam)

| Component | Accuracy |
|---|---|
| SELECT column correct | 29.37% |
| Aggregation correct | 88.24% |
| WHERE clause correct | 23.99% |

---

## Correctness Checks

All checks run with `model.eval()` and `torch.no_grad()`:

| Check | Result |
|---|---|
| Causal mask: max diff at earlier positions when last token changed | **0.0** ✅ |
| Padding mask: max diff when extra `<pad>` tokens added to source | **0.0** ✅ |
| Attention rows sum to 1 (all heads, all positions) | **min=1.0, max=1.0** ✅ |
| Weight sharing: `output.weight is embedding.weight` | **True** ✅ |
| LR schedule rises for 4000 steps then decays | **Verified** ✅ |

---

## Failure Cases

| # | Question | Gold SQL | Predicted SQL | Failure Type |
|---|---|---|---|---|
| 1 | What position does player #23 play? | SELECT Position WHERE No. = 23 | SELECT Player WHERE No. = 23 | Wrong column |
| 2 | How many gold medals did the USA win in total? | SELECT SUM(Gold) WHERE Country = USA | SELECT Gold WHERE Country = USA | Missing aggregation |
| 3 | Who scored more than 30 goals? | SELECT Player WHERE Goals > 30 | SELECT Player WHERE Goals = 30 | Wrong operator |
| 4 | What is the capital of the largest country? | SELECT Capital WHERE Area > ... | SELECT Capital | Missing WHERE clause |
| 5 | What is the average salary in engineering? | SELECT AVG(Salary) WHERE Dept = engineering | SELECT Salary WHERE Dept = engineering | Missing aggregation |
