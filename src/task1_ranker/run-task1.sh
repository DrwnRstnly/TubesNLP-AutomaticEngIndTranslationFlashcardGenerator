#!/bin/bash
set -e

# Folder output
MODEL_DIR="models"
RESULT_DIR="results"
mkdir -p $MODEL_DIR
mkdir -p $RESULT_DIR

echo "Starting fine tuning for all models..."

# Fine tune models
python3 finetune.py --model minilm_l6 --out ${MODEL_DIR}/minilm_l6_finetuned --epochs 5
python3 finetune.py --model minilm_l12 --out ${MODEL_DIR}/minilm_l12_finetuned --epochs 5
python3 finetune.py --model mpnet --out ${MODEL_DIR}/mpnet_finetuned --epochs 5
python3 finetune.py --model e5_base --out ${MODEL_DIR}/e5_base_finetuned --epochs 5
python3 finetune.py --model e5_large --out ${MODEL_DIR}/e5_large_finetuned --epochs 5

python3 finetune.py --model minilm_l6 --out ${MODEL_DIR}/minilm_l6_finetuned --epochs--lora 5
python3 finetune.py --model minilm_l12 --out ${MODEL_DIR}/minilm_l12_finetuned --epochs --lora 5
python3 finetune.py --model mpnet --out ${MODEL_DIR}/mpnet_finetuned --epochs --lora 5
python3 finetune.py --model e5_base --out ${MODEL_DIR}/e5_base_finetuned --epochs --lora 5
python3 finetune.py --model e5_large --out ${MODEL_DIR}/e5_large_finetuned --epochs --lora 5

echo "Fine tuning completed."
echo "Running evaluation..."

# # Evaluate both finetuned models and base models
python3 eval.py --models_dir ${MODEL_DIR} --out ${RESULT_DIR}/eval_finetuned_and_base.csv

echo "All done! Results saved in ${RESULT_DIR}/eval_finetuned_and_base.csv"
