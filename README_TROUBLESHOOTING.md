# IndexTTS2 模型加载问题诊断指南

## 问题描述

在服务器部署IndexTTS2时，执行启动命令后GPU占用5G内存，但模型未完全加载就卡住了。

## 可能的原因分析

### 1. 内存不足问题
- **现象**: GPU占用5G后卡住
- **原因**: IndexTTS2是一个大型多模态模型，需要大量GPU内存
- **完整模型大约需要**: 8-12GB GPU内存

### 2. 网络下载问题
- **现象**: 在加载某些组件时卡住
- **原因**: 模型会自动下载一些外部依赖:
  - `facebook/w2v-bert-2.0` (SeamlessM4T特征提取器)
  - `amphion/MaskGCT` (语义编解码器)
  - `funasr/campplus` (说话人识别模型)

### 3. GPU内存碎片化
- **现象**: 虽然总内存足够，但无法分配连续内存块
- **原因**: PyTorch内存分配策略问题

### 4. 依赖库版本冲突
- **现象**: 在加载transformers相关组件时失败
- **原因**: transformers版本与模型不兼容

## 解决方案

### 方案1: 内存优化配置

1. **修改环境变量** (已在更新的api_server.py中设置):
```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=1
```

2. **使用FP16精度** (已默认启用):
- 可以减少约50%的内存使用

3. **清理GPU缓存**:
```python
torch.cuda.empty_cache()
```

### 方案2: 离线模式部署

1. **预下载所有依赖模型**:
```bash
# 创建本地缓存目录
mkdir -p hf_cache/transformers

# 手动下载模型到本地
python -c "
from transformers import SeamlessM4TFeatureExtractor
SeamlessM4TFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
"
```

2. **设置离线环境变量**:
```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### 方案3: 分段加载

使用提供的诊断脚本来确定卡住的具体步骤:

```bash
# 运行诊断脚本
python debug_model_loading.py

# 或运行监控脚本
python monitor_init.py
```

### 方案4: 硬件检查

1. **检查GPU驱动**:
```bash
nvidia-smi
```

2. **检查CUDA版本兼容性**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

3. **内存测试**:
```bash
python -c "
import torch
x = torch.randn(1000, 1000).cuda()
print('GPU基本操作正常')
"
```

## 启动命令优化

### 原命令:
```bash
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

### 优化后的启动命令:
```bash
# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=4

# 启动服务
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

## 监控和日志

### 1. 使用增强的API服务器
更新后的`api_server.py`包含详细的初始化日志，会显示:
- 每个步骤的GPU内存使用
- 模型文件检查状态
- 加载时间统计

### 2. 实时监控
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控系统资源
htop
```

### 3. 日志分析
检查日志中的关键信息:
- GPU内存分配失败
- 网络连接超时
- 文件加载错误

## 常见错误解决

### 1. CUDA Out of Memory
```
解决方案:
- 减少batch_size
- 使用CPU模式: device="cpu"
- 升级GPU硬件
```

### 2. 网络连接超时
```
解决方案:
- 使用代理: export https_proxy=http://proxy:port
- 离线部署
- 手动下载模型文件
```

### 3. 依赖版本冲突
```
解决方案:
- 更新transformers: pip install transformers>=4.30.0
- 检查torch版本兼容性
- 重新创建虚拟环境
```

## 性能优化建议

1. **GPU内存管理**:
   - 使用渐进式加载
   - 实现模型分片

2. **网络优化**:
   - 使用本地模型缓存
   - 设置镜像源

3. **系统优化**:
   - 增加swap空间
   - 优化内核参数

## 调试步骤

1. **运行诊断脚本**:
```bash
python debug_model_loading.py
```

2. **检查系统状态**:
```bash
python monitor_init.py
```

3. **分析日志**:
- 查看GPU内存使用模式
- 确定卡住的具体步骤
- 检查网络连接状态

4. **逐步排查**:
- 先测试基本PyTorch功能
- 再测试单个模型组件
- 最后测试完整加载

通过以上方法，应该能够确定具体的问题原因并找到相应的解决方案。 