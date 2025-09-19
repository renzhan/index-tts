#!/bin/bash
# IndexTTS2 API Server 启动脚本

echo "启动IndexTTS2 API服务器..."

# 检查模型目录是否存在
if [ ! -d "models/IndexTTS-2" ]; then
    echo "错误：模型目录 models/IndexTTS-2 不存在"
    echo "请确保已下载IndexTTS2模型文件"
    exit 1
fi

# 检查必要的模型文件
required_files=("config.yaml" "gpt.pth" "s2mel.pth" "bpe.model" "wav2vec2bert_stats.pt" "feat1.pt" "feat2.pt")
for file in "${required_files[@]}"; do
    if [ ! -f "models/IndexTTS-2/$file" ]; then
        echo "错误：缺少必要文件 models/IndexTTS-2/$file"
        exit 1
    fi
done

echo "模型文件检查完成"

# 启动API服务器
echo "启动API服务器..."
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
