@echo off
echo ========================================================
echo       LoveTranscriber 安全启动器 (Ultra 9 专用)
echo ========================================================

:: 1. 禁用 OpenMP 的核心绑定 (防止大小核识别错误)
set KMP_AFFINITY=disabled

:: 2. 设置线程等待策略为被动 (防止 CPU 空转抢占)
set OMP_WAIT_POLICY=PASSIVE

:: 3. 强制 MKL 使用 AVX2 (防止新指令集崩溃)
set MKL_ENABLE_INSTRUCTIONS=AVX2

:: 4. 解决部分库冲突的万能药
set KMP_DUPLICATE_LIB_OK=TRUE

echo 正在启动...
echo 如果依然闪退，请将 crash.log 发给我。
echo --------------------------------------------------------

:: 启动程序
LoveTranscriber_Portable.exe

if %errorlevel% neq 0 (
    echo.
    echo 程序异常退出，错误代码：%errorlevel%
    pause
)