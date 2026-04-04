#!/bin/bash
# run_cross_dataset.sh
# DINOv2-UNet Cross-Dataset Evaluation & Joint Training Pipeline

# Exit on error
set -e

# ==============================================================================
# Step 1: Joint Training (Train out-of-domain features)
# Combine Kvasir and ClinicDB datasets to learn a universal representation.
# We use a 5-fold cross-validation setup, running just fold 0 here as an example.
# ==============================================================================

echo ">>> Starting Joint Training on Kvasir-SEG and CVC-ClinicDB (Fold 0)..."
python train.py \
    --dataset kvasir clinicdb \
    --data-dir ./data \
    --joint-train \
    --fold 0 \
    --num-folds 5 \
    --epochs 80 \
    --batch-size 8 \
    --save-dir runs/cross_dataset_experiment

# Given the dynamic naming, the model will be saved under:
# runs/cross_dataset_experiment/dinov2_unet_joint_kvasir_clinicdb/fold_0/best.pt

CHECKPOINT="runs/cross_dataset_experiment/dinov2_unet_joint_kvasir_clinicdb/fold_0/best.pt"

# ==============================================================================
# Step 2: Cross-Dataset Inference (Zero-shot Transfer)
# Test the independently trained models on entirely unseen datasets.
# ==============================================================================

echo ">>> Evaluating on ETIS-LaribPolypDB (Zero-shot generalization)..."
python test.py \
    --dataset etis \
    --data-dir ./data \
    --checkpoint $CHECKPOINT \
    --export-masks

echo ">>> Evaluating on CVC-ColonDB (Zero-shot generalization)..."
python test.py \
    --dataset colondb \
    --data-dir ./data \
    --checkpoint $CHECKPOINT \
    --export-masks

echo ">>> Cross-Dataset Evaluation Complete! Results saved to exports."
