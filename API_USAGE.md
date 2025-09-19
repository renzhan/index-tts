# IndexTTS2 API Server 使用指南

## 概述

IndexTTS2 API Server 是一个基于FastAPI的语音合成服务，兼容OpenAI的TTS接口，支持通过参考音频路径进行语音合成。

## 安装依赖

使用uv安装API服务所需的依赖：

```bash
# 安装API服务依赖
uv sync --extra api

# 或者安装所有可选功能（包括API、WebUI、DeepSpeed）
uv sync --all-extras
```

## 启动服务

### 方法1：使用启动脚本
```bash
./start_api.sh
```

### 方法2：使用uv运行（推荐）
```bash
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

### 方法3：直接使用Python
```bash
python3 api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

## API端点

### 1. 健康检查
```http
GET /health
```

响应：
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu",
    "timestamp": 1695123456.789
}
```

### 2. OpenAI兼容接口
```http
POST /v1/audio/speech
Content-Type: application/json

{
    "text": "大家好，欢迎使用IndexTTS2语音合成系统。",
    "voice": "/path/to/reference_audio.wav",
    "model": "indextts2",
    "response_format": "wav",
    "speed": 1.0
}
```

### 3. 完整功能接口
```http
POST /api/v1/tts
Content-Type: application/json

{
    "text": "大家好，欢迎使用IndexTTS2语音合成系统。",
    "voice": "/path/to/reference_audio.wav",
    "model": "indextts2",
    "response_format": "wav",
    "speed": 1.0,
    "emo_audio_prompt": "/path/to/emotion_audio.wav",
    "emo_alpha": 0.8,
    "emo_vector": [0, 0, 0, 0, 0, 0, 0.5, 0],
    "use_emo_text": false,
    "emo_text": "高兴",
    "use_random": false,
    "verbose": true
}
```

### 4. 文件上传接口
```http
POST /api/v1/tts/upload
Content-Type: multipart/form-data

text: 大家好，欢迎使用IndexTTS2语音合成系统。
voice_file: [音频文件]
emo_audio_file: [情感音频文件] (可选)
emo_alpha: 0.8
emo_vector: 0,0,0,0,0,0,0.5,0 (可选)
use_emo_text: false
emo_text: 高兴 (可选)
use_random: false
verbose: true
```

### 5. 列出模型
```http
GET /api/v1/models
```

### 6. 列出声音
```http
GET /api/v1/voices
```

## 参数说明

### 基本参数
- `text`: 要合成的文本
- `voice`: 参考音频文件路径
- `model`: 模型名称（默认：indextts2）
- `response_format`: 响应格式（默认：wav）
- `speed`: 语速倍数（默认：1.0）

### 情感控制参数
- `emo_audio_prompt`: 情感参考音频路径（可选）
- `emo_alpha`: 情感强度，范围0.0-1.0（默认：1.0）
- `emo_vector`: 情感向量，8个浮点数 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]（可选）
- `use_emo_text`: 是否使用文本情感分析（默认：false）
- `emo_text`: 情感描述文本（可选）
- `use_random`: 是否使用随机采样（默认：false）
- `verbose`: 是否输出详细信息（默认：false）

## 使用示例

### Python示例
```python
import requests
import json

# OpenAI兼容接口
url = "http://localhost:8000/v1/audio/speech"
data = {
    "text": "大家好，欢迎使用IndexTTS2语音合成系统。",
    "voice": "/path/to/reference_audio.wav",
    "model": "indextts2"
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)

# 完整功能接口
url = "http://localhost:8000/api/v1/tts"
data = {
    "text": "哇塞！这个爆率也太高了！欧皇附体了！",
    "voice": "/path/to/reference_audio.wav",
    "emo_vector": [0, 0, 0, 0, 0, 0, 0.45, 0],
    "use_random": False,
    "verbose": True
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL示例
```bash
# OpenAI兼容接口
curl -X POST "http://localhost:8000/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "大家好，欢迎使用IndexTTS2语音合成系统。",
       "voice": "/path/to/reference_audio.wav",
       "model": "indextts2"
     }' \
     --output output.wav

# 文件上传接口
curl -X POST "http://localhost:8000/api/v1/tts/upload" \
     -F "text=大家好，欢迎使用IndexTTS2语音合成系统。" \
     -F "voice_file=@/path/to/reference_audio.wav" \
     -F "emo_alpha=0.8" \
     -F "verbose=true" \
     --output output.wav
```

## 错误处理

API会返回标准的HTTP状态码：

- `200`: 成功
- `400`: 请求参数错误
- `503`: 模型未加载
- `500`: 服务器内部错误

错误响应格式：
```json
{
    "detail": "错误描述"
}
```

## 注意事项

1. 确保参考音频文件存在且为WAV格式
2. 情感向量必须包含8个浮点数
3. 文件上传接口会自动清理临时文件
4. 建议使用FP16模式以节省显存
5. 支持CORS，可以跨域访问

## 开发模式

启动开发模式（自动重载）：
```bash
uv run api_server.py --reload
```

## 测试方法

API依赖已经包含在 `--extra api` 中，无需额外安装。

运行测试：
```bash
# 测试所有功能（需要提供参考音频文件）
uv run test_api.py --voice-path /path/to/your/audio.wav

# 只测试健康检查
uv run test_api.py --test-health --voice-path /path/to/your/audio.wav

# 只测试OpenAI兼容接口
uv run test_api.py --test-openai --voice-path /path/to/your/audio.wav

# 只测试完整功能接口
uv run test_api.py --test-full --voice-path /path/to/your/audio.wav

# 只测试文件上传接口
uv run test_api.py --test-upload --voice-path /path/to/your/audio.wav
```

## 性能优化

1. 使用FP16模式：`use_fp16=True`
2. 使用CUDA内核：`use_cuda_kernel=True`（需要GPU）
3. 调整工作进程数：`--workers 4`
4. 使用反向代理（如Nginx）进行负载均衡
