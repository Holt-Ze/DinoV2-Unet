@echo off
REM ============================================================================
REM DINOv2-UNet: 辅助工具脚本
REM 环境检查、数据验证、日志清理
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

:menu
cls
echo.
echo ============================================================================
echo  DINOv2-UNet 辅助工具
echo ============================================================================
echo.
echo  1. 环境检查    - 检查Python、PyTorch、CUDA是否正确配置
echo  2. 数据验证    - 验证数据集完整性和格式
echo  3. 日志清理    - 清理旧的运行日志和缓存
echo  4. 查看上次运行 - 查看最近的实验结果
echo  5. 磁盘空间    - 检查可用空间和各目录大小
echo  0. 退出
echo.
echo ============================================================================
echo.

set /p choice="请选择操作 [0-5]: "

if "%choice%"=="0" goto exit
if "%choice%"=="1" goto check_env
if "%choice%"=="2" goto validate_data
if "%choice%"=="3" goto cleanup
if "%choice%"=="4" goto view_latest
if "%choice%"=="5" goto check_disk

echo [错误] 无效选择！
timeout /t 2 >nul
goto menu

REM ============================================================================
REM 1. 环境检查
REM ============================================================================
:check_env
cls
echo.
echo ============================================================================
echo  1. 环境检查
echo ============================================================================
echo.

echo [检查] Python 版本...
python --version
if errorlevel 1 (
    echo [错误] Python 未安装或不在 PATH 中！
    goto check_env_end
)

echo.
echo [检查] PyTorch 安装...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
if errorlevel 1 (
    echo [警告] PyTorch 未安装或加载失败
    goto check_env_end
)

echo.
echo [检查] timm 库...
python -c "import timm; print(f'timm version: {timm.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo [警告] timm 库未安装
)

echo.
echo [检查] 项目模块...
python -c "import seg; print('[√] seg 包可以导入')" >nul 2>&1
if errorlevel 1 (
    echo [警告] seg 包导入失败，可能需要重新安装依赖
)

echo.
echo [检查] 数据集目录...
if exist ../data\ (
    echo [√] data/ 目录存在
    dir /b ../data\ | findstr /r "Kvasir CVC ETIS" >nul
    if errorlevel 1 (
        echo [警告] data/ 目录中未找到预期的数据集文件夹
    ) else (
        echo [√] 发现数据集文件夹
    )
) else (
    echo [警告] data/ 目录不存在，请创建并放入数据集
)

:check_env_end
echo.
echo 环境检查完毕。按任意键返回菜单...
pause >nul
goto menu

REM ============================================================================
REM 2. 数据验证
REM ============================================================================
:validate_data
cls
echo.
echo ============================================================================
echo  2. 数据验证
echo ============================================================================
echo.

if not exist ../data\ (
    echo [错误] data/ 目录不存在！
    echo 请先下载数据集：
    echo   - Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
    echo   - CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/
    echo   - CVC-ColonDB: http://vi.cvc.uab.es/colon-qa/cvccolondb/
    echo   - ETIS: http://vi.cvc.uab.es/colon-qa/cvccolondb/
    goto validate_data_end
)

echo [检查] Kvasir-SEG...
if exist ../data\Kvasir-SEG\images\ if exist ../data\Kvasir-SEG\masks\ (
    for /f %%i in ('dir /b ../data\Kvasir-SEG\images\ ^| find /c /v ""') do set count=%%i
    echo [√] Kvasir-SEG: !count! 张图像
) else (
    echo [✗] Kvasir-SEG: 缺少 images/ 或 masks/ 文件夹
)

echo [检查] CVC-ClinicDB...
if exist ../data\CVC-ClinicDB\Original\ if exist ../data\CVC-ClinicDB\Ground\ (
    for /f %%i in ('dir /b ../data\CVC-ClinicDB\Original\ ^| find /c /v ""') do set count=%%i
    echo [√] CVC-ClinicDB: !count! 张图像
) else (
    echo [✗] CVC-ClinicDB: 缺少 Original/ 或 Ground Truth/ 文件夹
)

echo [检查] CVC-ColonDB...
if exist ../data\CVC-ColonDB\images\ if exist ../data\CVC-ColonDB\masks\ (
    for /f %%i in ('dir /b ../data\CVC-ColonDB\images\ ^| find /c /v ""') do set count=%%i
    echo [√] CVC-ColonDB: !count! 张图像
) else (
    echo [✗] CVC-ColonDB: 缺少 images/ 或 masks/ 文件夹
)

echo [检查] ETIS...
if exist ../data\ETIS\images\ if exist ../data\ETIS\masks\ (
    for /f %%i in ('dir /b ../data\ETIS\images\ ^| find /c /v ""') do set count=%%i
    echo [√] ETIS: !count! 张图像
) else (
    echo [✗] ETIS: 缺少 images/ 或 masks/ 文件夹
)

:validate_data_end
echo.
echo 数据验证完毕。按任意键返回菜单...
pause >nul
goto menu

REM ============================================================================
REM 3. 日志清理
REM ============================================================================
:cleanup
cls
echo.
echo ============================================================================
echo  3. 日志清理
echo ============================================================================
echo.

echo [提示] 这将删除以下内容：
echo   • log/ 目录（训练日志）
echo   • __pycache__/ 目录（Python缓存）
echo   • *.pyc 文件（编译的Python字节码）
echo.

set /p confirm="是否继续？(y/n): "
if /i not "%confirm%"=="y" (
    echo 已取消。
    goto cleanup_end
)

echo.
echo [清理] 删除log目录...
if exist ../log\ (
    rmdir /s /q ../log\
    echo [√] log/ 已删除
)

echo [清理] 删除Python缓存...
for /d /r .. %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
    )
)
echo [√] __pycache__/ 已删除

echo [清理] 删除.pyc文件...
for /r .. %%f in (*.pyc) do (
    if exist "%%f" del "%%f"
)
echo [√] *.pyc 已删除

echo.
echo [完成] 清理完毕！

:cleanup_end
echo.
echo 按任意键返回菜单...
pause >nul
goto menu

REM ============================================================================
REM 4. 查看上次运行
REM ============================================================================
:view_latest
cls
echo.
echo ============================================================================
echo  4. 查看上次运行结果
echo ============================================================================
echo.

if not exist ../runs\ (
    echo [提示] runs/ 目录不存在，还没有训练过模型。
    goto view_latest_end
)

echo [最近的训练运行]
echo.
for /d %%d in (../runs\*) do (
    if exist "%%d\best.pt" (
        echo 目录: %%~nd
        if exist "%%d\metrics_history.json" (
            echo   [✓] 有指标记录
        )
        if exist "%%d\failure_analysis.json" (
            echo   [✓] 有失败分析
        )
        if exist "%%d\failure_montages\" (
            echo   [✓] 有失败可视化
        )
    )
)

echo.
echo [消融研究结果]
if exist ../ablation_results\summary.csv (
    echo   [✓] ablation_results\summary.csv
) else (
    echo   [✗] 未找到消融结果
)

echo.
echo [实验报告]
if exist ../reports\experiment_report.html (
    echo   [✓] reports\experiment_report.html
    echo   可以用浏览器打开查看完整报告
    set /p open="是否用默认浏览器打开？(y/n): "
    if /i "%open%"=="y" (
        start ../reports\experiment_report.html
    )
) else (
    echo   [✗] 未找到实验报告
)

:view_latest_end
echo.
echo 按任意键返回菜单...
pause >nul
goto menu

REM ============================================================================
REM 5. 磁盘空间
REM ============================================================================
:check_disk
cls
echo.
echo ============================================================================
echo  5. 磁盘空间检查
echo ============================================================================
echo.

echo [系统磁盘空间]
wmic logicaldisk where name="C:" get freespace,size
echo.

echo [项目目录大小估计]
echo   • runs/ - 模型和日志
echo   • ablation_results/ - 消融研究结果
echo   • reports/ - 实验报告
echo   • log/ - 训练日志
echo.

echo [建议]
echo   • 训练一个模型需要 ~1-2 GB
echo   • 完整消融需要 ~10-20 GB
echo   • 高级管道需要 ~30-50 GB
echo.
echo 如果磁盘空间不足，可以运行"日志清理"来释放空间。

echo.
echo 按任意键返回菜单...
pause >nul
goto menu

:exit
echo.
echo 已退出。
echo.
