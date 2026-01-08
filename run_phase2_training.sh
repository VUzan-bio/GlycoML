#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "PHASE 2 TRAINING PIPELINE (BASELINE)"
echo "=========================================="

echo ""
echo "[1/5] Merging datasets..."
python scripts/data/merge_phase2_datasets.py

echo ""
echo "[2/5] Preprocessing glycan structures..."
python scripts/data/preprocess_glycan_structures.py

echo ""
echo "[3/5] Training glycan embeddings (baseline SVD)..."
python scripts/baselines/train_phase2_glycan_gnn.py --epochs 50 --batch-size 32 --learning-rate 0.001

echo ""
echo "[4/5] Training binding predictor (baseline logistic regression)..."
python scripts/baselines/train_phase2_lectin_glycan_model.py --epochs 100 --batch-size 64 --learning-rate 0.001 --use-glycan-embeddings

echo ""
echo "[5/5] Evaluating model..."
python scripts/analysis/evaluate_phase2_model.py

echo ""
echo "=========================================="
echo "PHASE 2 BASELINE TRAINING COMPLETE"
echo "=========================================="
