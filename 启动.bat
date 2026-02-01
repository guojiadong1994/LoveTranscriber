@echo off
echo ========================================================
echo       正在为 Intel Ultra 9 配置安全环境...
echo ========================================================

:: 1. 强制 MKL 数学库降级到 AVX2 指令集 (核心救命药)
set MKL_ENABLE_INSTRUCTIONS=AVX2

:: 2. 开启 CTranslate2 的实验性 GEMM 支持 (官方推荐的 Windows 补丁)
set CT2_USE_EXPERIMENTAL_PACKED_GEMM=1

:: 3. 限制 OpenMP 线程数，防止大小核死锁
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE

echo 环境配置完毕，正在启动软件...
echo 请勿关闭此黑框，如果闪退请截图！
echo --------------------------------------------------------

:: 启动你的程序 (注意：名字要和你打包出来的 exe 名字一致)
LoveTranscriber_Portable.exe

:: 如果程序意外结束，暂停显示报错
if %errorlevel% neq 0 (
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo 程序发生崩溃！错误代码：%errorlevel%
    echo 请把上面的报错信息（如果有）截图。
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pause
)