@echo off
REM ============================================================================
REM DINOv2-UNet 一键工具集 - 总启动菜单
REM 所有实验和管理工具的统一入口
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

:main_menu
cls
echo.
echo ============================================================================
echo             DINOv2-UNet 一键工具集 - 总启动菜单
echo ============================================================================
echo.
echo 【实验管道】
echo   1. 快速管道     (20-30分钟)   - 快速验证代码和数据
echo   2. 完整管道⭐  (2-4小时)     - 全面实验分析(推荐)
echo   3. 高级管道     (6-10小时)    - 多数据集联合训练+跨域评估
echo   4. 管道启动器   - 交互式菜单选择管道
echo.
echo 【辅助工具】
echo   5. 环境检查     - 检查PyTorch、CUDA、GPU
echo   6. 数据验证     - 验证数据集完整性
echo   7. 查看结果     - 浏览上次运行的结果
echo   8. 日志清理     - 清理旧日志和缓存
echo   9. 打开工具箱   - 所有辅助工具菜单
echo.
echo 【文档】
echo   H. 打开使用指南 - BATCH_SCRIPTS_GUIDE.md
echo   R. 打开README   - 项目说明文档
echo.
echo   0. 退出
echo.
echo ============================================================================
echo.

set /p choice="请输入选项 [0-9, H, R]: "

if /i "%choice%"=="0" goto exit
if /i "%choice%"=="1" goto quick
if /i "%choice%"=="2" goto full
if /i "%choice%"=="3" goto advanced
if /i "%choice%"=="4" goto launcher
if /i "%choice%"=="5" goto check_env
if /i "%choice%"=="6" goto validate
if /i "%choice%"=="7" goto view_results
if /i "%choice%"=="8" goto cleanup
if /i "%choice%"=="9" goto tools
if /i "%choice%"=="H" goto guide
if /i "%choice%"=="R" goto readme

echo [错误] 无效选项，请重新输入！
timeout /t 2 >nul
goto main_menu

REM ============================================================================
REM 实验管道
REM ============================================================================
:quick
cls
echo 启动快速管道...
timeout /t 1 >nul
call run_quick_pipeline.bat
goto back_to_menu

:full
cls
echo 启动完整管道...
timeout /t 1 >nul
call run_full_pipeline.bat
goto back_to_menu

:advanced
cls
echo 启动高级管道...
timeout /t 1 >nul
call run_advanced_pipeline.bat
goto back_to_menu

:launcher
cls
call run_pipeline.bat
goto back_to_menu

:tools
cls
call run_tools.bat
goto back_to_menu

REM ============================================================================
REM 快捷工具
REM ============================================================================
:check_env
cls
echo.
echo [环境检查] Python版本...
python --version
echo.
echo [PyTorch检查]...
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
echo.
echo 按任意键返回菜单...
pause >nul
goto main_menu

:validate
cls
echo.
echo [数据集检查]
echo.
if exist ..\data\Kvasir-SEG\images\ (
    for /f %%i in ('dir /b ..\data\Kvasir-SEG\images\ 2^>nul ^| find /c /v ""') do echo [√] Kvasir-SEG: %%i 张图像
) else (
    echo [✗] Kvasir-SEG: 未找到
)
if exist ..\data\CVC-ClinicDB\Original\ (
    for /f %%i in ('dir /b ..\data\CVC-ClinicDB\Original\ 2^>nul ^| find /c /v ""') do echo [√] CVC-ClinicDB: %%i 张图像
) else (
    echo [✗] CVC-ClinicDB: 未找到
)
if exist ..\data\CVC-ColonDB\images\ (
    for /f %%i in ('dir /b ..\data\CVC-ColonDB\images\ 2^>nul ^| find /c /v ""') do echo [√] CVC-ColonDB: %%i 张图像
) else (
    echo [✗] CVC-ColonDB: 未找到
)
if exist ..\data\ETIS\images\ (
    for /f %%i in ('dir /b ..\data\ETIS\images\ 2^>nul ^| find /c /v ""') do echo [√] ETIS: %%i 张图像
) else (
    echo [✗] ETIS: 未找到
)
echo.
echo 按任意键返回菜单...
pause >nul
goto main_menu

:view_results
cls
echo.
echo [最近的运行结果]
echo.
if exist ..\runs\ (
    echo 发现 runs/ 目录
    if exist ..\reports\experiment_report.html (
        set /p open="发现实验报告，是否用浏览器打开？(y/n): "
        if /i "!open!"=="y" start ..\reports\experiment_report.html
    )
    if exist ..\ablation_results\summary.csv (
        set /p open="发现消融结果，是否用记事本打开？(y/n): "
        if /i "!open!"=="y" start ..\ablation_results\summary.csv
    )
) else (
    echo [提示] runs/ 目录不存在，还没有训练过。
)
echo.
echo 按任意键返回菜单...
pause >nul
goto main_menu

:cleanup
cls
echo.
echo [清理日志和缓存]
echo.
set /p confirm="确认删除 log/ 和 __pycache__？(y/n): "
if /i not "!confirm!"=="y" (
    echo 已取消。
) else (
    if exist ..\log\ rmdir /s /q ..\log\ 2>nul
    for /d /r .. %%d in (__pycache__) do rmdir /s /q "%%d" 2>nul
    echo [完成] 清理完毕
)
echo.
echo 按任意键返回菜单...
pause >nul
goto main_menu

REM ============================================================================
REM 文档
REM ============================================================================
:guide
if exist BATCH_SCRIPTS_GUIDE.md (
    start BATCH_SCRIPTS_GUIDE.md
) else (
    echo [错误] 找不到 BATCH_SCRIPTS_GUIDE.md
)
timeout /t 1 >nul
goto main_menu

:readme
if exist ..\README.md (
    start ..\README.md
) else (
    echo [错误] 找不到 README.md
)
timeout /t 1 >nul
goto main_menu

REM ============================================================================
REM 返回菜单
REM ============================================================================
:back_to_menu
timeout /t 1 >nul
goto main_menu

:exit
cls
echo.
echo 感谢使用 DINOv2-UNet 一键工具集！
echo.
echo 祝你实验顺利! 🎉
echo.
pause
