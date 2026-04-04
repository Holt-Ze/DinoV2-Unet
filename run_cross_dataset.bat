@echo off
REM run_cross_dataset.bat
REM DINOv2-UNet Cross-Dataset Evaluation & Joint Training Pipeline for Windows

REM ==============================================================================
REM Step 1: Joint Training (Train out-of-domain features)
REM Combine Kvasir and ClinicDB datasets to learn a universal representation.
REM We use a 5-fold cross-validation setup, running just fold 0 here as an example.
REM ==============================================================================

echo ^>^>^> Starting Joint Training on Kvasir-SEG and CVC-ClinicDB (Fold 0)...
python train.py ^
    --dataset kvasir clinicdb ^
    --data-dir ./data ^
    --joint-train ^
    --fold 0 ^
    --num-folds 5 ^
    --epochs 80 ^
    --batch-size 8 ^
    --save-dir runs/cross_dataset_experiment

IF %ERRORLEVEL% NEQ 0 (
    echo [Error] Training failed!
    exit /b %ERRORLEVEL%
)

REM Given the dynamic naming, the model will be saved under:
REM runs\cross_dataset_experiment\dinov2_unet_joint_kvasir_clinicdb\fold_0\best.pt
SET CHECKPOINT="runs\cross_dataset_experiment\dinov2_unet_joint_kvasir_clinicdb\fold_0\best.pt"

REM ==============================================================================
REM Step 2: Cross-Dataset Inference (Zero-shot Transfer)
REM Test the independently trained models on entirely unseen datasets.
REM ==============================================================================

echo ^>^>^> Evaluating on ETIS-LaribPolypDB (Zero-shot generalization)...
python test.py ^
    --dataset etis ^
    --data-dir ./data/ETIS ^
    --checkpoint %CHECKPOINT% ^
    --export-masks

echo ^>^>^> Evaluating on CVC-ColonDB (Zero-shot generalization)...
python test.py ^
    --dataset colondb ^
    --data-dir ./data/CVC-ColonDB ^
    --checkpoint %CHECKPOINT% ^
    --export-masks

echo ^>^>^> Cross-Dataset Evaluation Complete! Results saved to exports.
pause
