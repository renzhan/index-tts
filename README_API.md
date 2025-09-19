# IndexTTS2 API Server 安装和使用

## 快速开始

### 1. 安装依赖

```bash
# 安装API服务依赖
uv sync --extra api

# 或者安装所有可选功能（包括API、WebUI、DeepSpeed）
uv sync --all-extras
```

### 2. 启动服务

```bash
# 使用启动脚本（推荐）
./start_api.sh

# 或直接使用uv运行
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

### 3. 测试服务

```bash
# 测试所有功能（需要提供参考音频文件）
uv run test_api.py --voice-path /path/to/your/audio.wav

# 只测试健康检查
uv run test_api.py --test-health --voice-path /path/to/your/audio.wav
```

### 4. 访问API文档

打开浏览器访问：`http://localhost:8000/docs`

## 主要特性

- ✅ 兼容OpenAI TTS接口 (`/v1/audio/speech`)
- ✅ 支持参考音频路径 (`voice` 参数)
- ✅ 支持文件上传
- ✅ 支持情感控制
- ✅ 完善的错误处理
- ✅ CORS支持
- ✅ 自动文档生成

## API端点

- `GET /health` - 健康检查
- `POST /v1/audio/speech` - OpenAI兼容接口
- `POST /api/v1/tts` - 完整功能接口
- `POST /api/v1/tts/upload` - 文件上传接口
- `GET /api/v1/models` - 模型列表
- `GET /api/v1/voices` - 声音列表

## 使用示例

```python
import requests

# OpenAI兼容接口
response = requests.post("http://localhost:8000/v1/audio/speech", json={
    "text": "大家好，欢迎使用IndexTTS2语音合成系统。",
    "voice": "/path/to/reference_audio.wav",
    "model": "indextts2"
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

详细文档请参考：`API_USAGE.md`
