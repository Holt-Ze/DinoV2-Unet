@echo off
setlocal enabledelayedexpansion

:START
echo.
echo ============================================================================
echo  DINOv2-UNet: One-Click Experiment Pipeline
echo ============================================================================
echo.
echo Choose your experiment mode:
echo.
echo  1. Quick Pipeline    - Basic training with metrics tracking
echo  2. Full Pipeline     - Training + ablation + failure analysis
echo  3. Advanced Pipeline - Full + domain analysis + report
echo  4. Manual Mode       - Choose custom scripts from menu
echo  5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    goto quick
) else if "%choice%"=="2" (
    goto full
) else if "%choice%"=="3" (
    goto advanced
) else if "%choice%"=="4" (
    goto menu
) else if "%choice%"=="5" (
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    timeout /t 2 /nobreak
    cls
    goto START
)

:quick
echo.
echo Starting Quick Pipeline...
pushd ..
call scripts\run_quick_pipeline.bat
popd
goto end

:full
echo.
echo Starting Full Pipeline...
pushd ..
call scripts\run_full_pipeline.bat
popd
goto end

:advanced
echo.
echo Starting Advanced Pipeline...
pushd ..
call scripts\run_advanced_pipeline.bat
popd
goto end

:menu
cls
echo.
echo ============================================================================
echo  Manual Script Selection
echo ============================================================================
echo.
echo  1. Train with metrics tracking
echo  2. Run ablation studies
echo  3. Generate failure analysis
echo  4. Generate reports
echo  5. Run custom pipeline
echo  6. Back to main menu
echo.

set /p script_choice="Enter your choice (1-6): "

if "%script_choice%"=="1" (
    pushd ..
    python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG --track-metrics
    popd
) else if "%script_choice%"=="2" (
    pushd ..
    python run_ablation_studies.py --dataset kvasir --axes freeze_blocks_until lr_ratio
    popd
) else if "%script_choice%"=="3" (
    pushd ..
    python train.py --dataset kvasir --data-dir ./data/Kvasir-SEG --save-failure-analysis
    popd
) else if "%script_choice%"=="4" (
    pushd ..
    python generate_experiment_report.py --results-dir runs --output-dir reports
    popd
) else if "%script_choice%"=="5" (
    pushd ..
    call scripts\run_tools.bat
    popd
) else if "%script_choice%"=="6" (
    goto START
) else (
    echo Invalid choice.
    timeout /t 2 /nobreak
    goto menu
)

:end
echo.
echo ============================================================================
echo  Complete!
echo ============================================================================
echo.
echo Results saved to:
echo   - Models:      runs/
echo   - Ablations:   ablation_results/
echo   - Reports:     reports/
echo.
pause
