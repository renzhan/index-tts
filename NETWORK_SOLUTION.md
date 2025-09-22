# IndexTTS2 网络卡顿问题解决方案

## 问题确认

从你的日志可以看出，程序在以下位置卡住：
```
>> GPT weights restored from: models/IndexTTS-2/gpt.pth
2025-09-22 07:21:19,176 - indextts.gpt.transformers_modeling_utils - WARNING - GPT2InferenceModel has generative capabilities...
```

**卡住位置**: 在GPT模型加载完成后，正在尝试下载 `facebook/w2v-bert-2.0` (SeamlessM4T特征提取器)

## 立即解决方案

### 方案1: 使用HuggingFace镜像源 ⭐ 推荐

```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 然后重新启动
CUDA_VISIBLE_DEVICES=1 ./start_api.sh
```

### 方案2: 预下载所有模型

```bash
# 先运行预下载脚本
python predownload_models.py

# 下载完成后再启动
CUDA_VISIBLE_DEVICES=1 ./start_api.sh
```

### 方案3: 使用综合解决脚本

```bash
# 运行交互式解决方案
chmod +x solve_network_issue.sh
./solve_network_issue.sh
```

### 方案4: 设置代理（如果你有代理）

```bash
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
CUDA_VISIBLE_DEVICES=1 ./start_api.sh
```

### 方案5: 增加超时时间

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10分钟
export TRANSFORMERS_TIMEOUT=600
CUDA_VISIBLE_DEVICES=1 ./start_api.sh
```

## 诊断工具

### 1. 网络连接诊断
```bash
python network_debug.py
```

### 2. 快速检查
```bash
python quick_check.py
```

### 3. 测试导入
```bash
python test_imports.py
```

## 需要下载的模型列表

IndexTTS2 会自动下载以下模型：

1. **facebook/w2v-bert-2.0** (SeamlessM4T特征提取器) - 约2GB
2. **amphion/MaskGCT** (语义编解码器) - 约500MB  
3. **funasr/campplus** (说话人识别) - 约50MB
4. **BigVGAN声码器** (根据配置) - 约100MB

总计约 **2.7GB** 需要下载

## 离线部署方案

如果网络实在不稳定，可以考虑离线部署：

### 步骤1: 在有网络的机器上预下载
```bash
python predownload_models.py
```

### 步骤2: 打包缓存目录
```bash
# 查找缓存目录
python -c "from transformers.utils import default_cache_path; print(default_cache_path)"

# 打包缓存
tar -czf huggingface_cache.tar.gz ~/.cache/huggingface/
```

### 步骤3: 在目标服务器上解包
```bash
# 解包到相同位置
tar -xzf huggingface_cache.tar.gz -C ~/

# 设置离线模式
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 启动服务
CUDA_VISIBLE_DEVICES=1 ./start_api.sh
```

## 常见问题

### Q: 为什么会卡在这里？
A: IndexTTS2需要下载几个外部模型，网络不稳定时会无限等待

### Q: 可以跳过这些下载吗？
A: 不能，这些模型是必需的组件

### Q: 下载需要多长时间？
A: 取决于网络速度，通常5-30分钟

### Q: 可以用CPU模式避免这个问题吗？
A: 可以，但仍需要下载模型，只是内存占用更少

## 终极解决方案

如果所有方案都不行，可以：

1. **修改代码添加重试机制**
2. **使用本地模型文件替换**
3. **搭建本地HuggingFace镜像**

## 监控下载进度

可以用以下命令监控下载：

```bash
# 监控网络
watch -n 1 "ss -tuln | grep :443"

# 监控缓存目录
watch -n 5 "du -sh ~/.cache/huggingface/"

# 监控GPU内存
watch -n 1 nvidia-smi
```

## 立即行动

**推荐执行顺序**:

1. 首先尝试方案1 (HuggingFace镜像)
2. 如果不行，运行方案3 (综合解决脚本)  
3. 最后考虑离线部署

```bash
# 立即执行
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 ./start_api.sh
```

这样应该能解决你的网络卡顿问题！ 