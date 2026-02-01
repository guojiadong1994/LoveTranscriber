@echo off
echo ========================================================
echo       LoveTranscriber 安全启动器 (Ultra 9 专用)
echo ========================================================

:: 1) Intel OpenMP：禁用亲和性绑定（核心项）
set KMP_AFFINITY=disabled

:: 2) 等待策略/阻塞：降低初始化阶段风险
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0

:: 3) 初始化阶段强制单线程（核心项）
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1

:: 4) 指令集降级（双保险）
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
