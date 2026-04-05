@echo off
REM ============================================================================
REM DINOv2-UNet: 一键实验管道启动器
REM 选择要运行的实验配置
REM ============================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

:menu
cls
echo.
echo ============================================================================
echo  DINOv2-UNet 一键实验管道启动器
echo ============================================================================
echo.
echo  1. 快速管道 (快速) - 训练 + 简化消融 + 报告
echo     • 约20-30分钟
echo     • 适合快速验证
echo.
echo  2. 完整管道 (标准) - 训练 + 完整消融 + 失败分析 + 报告
echo     • 约2-4小时
echo     • 推荐用于发表研究
echo.
echo  3. 高级管道 (全面) - 联合训练 + 交叉验证 + 零样本评估 + 消融 + 报告
echo     • 约6-10小时
echo     • 最完整的实验分析
echo.
echo  0. 退出
echo.
echo ============================================================================
echo.

set /p choice="请选择运行的管道 [0-3]: "

if "%choice%"=="0" goto exit
if "%choice%"=="1" goto quick
if "%choice%"=="2" goto full
if "%choice%"=="3" goto advanced

echo [错误] 无效选择，请重新输入！
timeout /t 2 >nul
goto menu

:quick
echo.
echo 启动快速管道...
echo.
call run_quick_pipeline.bat
goto end

:full
echo.
echo 启动完整管道...
echo.
call run_full_pipeline.bat
goto end

:advanced
echo.
echo 启动高级管道...
echo.
call run_advanced_pipeline.bat
goto end

:end
echo.
echo 管道脚本已执行完毕！
echo.
goto menu

:exit
echo.
echo 已退出。
echo.
