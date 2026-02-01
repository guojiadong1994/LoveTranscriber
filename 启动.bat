@echo off
echo ========================================================
echo       LoveTranscriber 安全启动器 (Ultra 9 专用)
echo ========================================================

set KMP_AFFINITY=disabled
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0
set OMP_PROC_BIND=FALSE
set OMP_PLACES=cores
set OMP_DYNAMIC=FALSE

set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_ENABLE_INSTRUCTIONS=AVX2

echo 正在启动...
echo --------------------------------------------------------
LoveTranscriber_Portable.exe

if %errorlevel% neq 0 (
    echo.
    echo 程序异常退出，错误代码：%errorlevel%
    echo 请将 crash.log 发给我。
    pause
)
