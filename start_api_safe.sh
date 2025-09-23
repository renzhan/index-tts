#!/bin/bash

# IndexTTS2 API服务器安全启动脚本
# 此脚本禁用CUDA内核编译，避免环境问题导致卡住

echo "================================"
echo "IndexTTS2 安全启动模式"
echo "================================"

echo "注意：此模式禁用CUDA内核编译以避免环境问题"
echo "性能可能稍有下降，但功能完全正常"
echo ""

# 设置环境变量
export DISABLE_CUDA_KERNEL=1
export TORCH_CUDA_ARCH_LIST=""

# 检查必要文件
if [ ! -f "models/IndexTTS-2/config.yaml" ]; then
    echo "错误：未找到模型配置文件 models/IndexTTS-2/config.yaml"
    echo "请确保模型已正确下载到 models/IndexTTS-2/ 目录"
    exit 1
fi

echo "启动参数说明："
echo "  --host 0.0.0.0        : 允许外部访问"
echo "  --port 8000           : 监听端口8000"
echo "  --model-dir models/IndexTTS-2 : 模型目录"
echo "  --disable-cuda-kernel : 禁用CUDA内核编译"
echo ""

echo "正在启动API服务器..."
echo "如需停止服务，请按 Ctrl+C"
echo ""

# 启动API服务器
uv run api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model-dir models/IndexTTS-2 \
    --disable-cuda-kernel

echo ""
echo "API服务器已停止" 