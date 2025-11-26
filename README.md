# Automatic Flashcard Generator

An English-Indonesian learning flashcards from raw text using a LangGraph pipeline.  
The main entrypoint is `main.py`, which exposes a `/generate` endpoint that orchestrates the pipeline defined in `src/pipeline/langgraph_pipeline.py`.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Datasets](#datasets)

---

## Overview

This API receives a block of text and an extraction prompt, then runs a pipeline to:
- Select top-𝑘 representative sentences
- Classify the communicative intent of each sentence
- Estimate difficulty (e.g., CEFR A1–C2)
- Generate translations (into Indonesian)

and returns all of this as a structured JSON response suitable for building flashcard UIs.

---

## Tech Stack
- **Python**
- **FastAPI**
- **Pydantic**
- **LangGraph pipeline**

---

## Project Structure
```
.
├── notebooks/
│   ├── 01_task1_ranker_eda.ipynb
│   ├── 01_task1_ranker_eval.ipynb
│   ├── 02_task2_intent.ipynb
│   ├── 03_task3_cefr_eda.ipynb
│   ├── 03_task3_cefr_eval.ipynb
│   ├── 04_task4_translation.ipynb
│   └── 04_task4_visual.ipynb
├── src/
│   ├── api/
│   │   └── main.py
│   ├── pipeline/
│   │   ├── langgraph_pipeline.py
│   │   └── state.py
│   ├── task1_ranker/
│   │   ├── results
│   │   ├── test
│   │   ├── eval.py
│   │   ├── finetune.py
│   │   ├── inference.py
│   │   ├── node.py
│   │   └── run-task1.sh
│   ├── task2_intent/
│   │   ├── inference.py
│   │   └── node.py
│   ├── task3_cefr/
│   │   ├── results
│   │   ├── test
│   │   ├── eval.py
│   │   ├── inference.py
│   │   └── node.py
│   └── task4_mt/
│       ├── Experiments
│       ├── inference.py
│       └── node.py
├── README.md
└── requirements.txt
```

---

## Installation
```
pip install -r requirements.txt
```

---

## Running the Server
```
uvicorn src.api.main:app --reload
```

---

## Datasets
1. Dataset Task 1: https://huggingface.co/datasets/mteb/stsbenchmark-sts
2. Dataset Task 2: https://huggingface.co/datasets/ConvLab/dailydialog
3. Dataset Task 3: https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en
4. Dataset Task 4: https://drive.google.com/drive/folders/1YWx21zOhW086bXm3QMF6pIh1IKqUP9Zo?usp=sharing

Informasi Tambahan Task 4:
1. aligned_translation_pairs -> Data dari TEDx
2. english.txt -> Data Flores Bahasa Inggris
3. indo.txt -> Data Flores Bahasa Inggris

---