# IndexTTS2 模型初始化日志增强

## 概述
为了诊断BigVGAN模型加载失败的问题，我们为IndexTTS2的初始化过程增加了详细的日志记录。

## 增强的日志功能

### 1. API服务器日志增强 (`api_server.py`)
- 创建了`IndexTTS2WithLogging`包装类
- 捕获所有初始化过程中的print输出并转换为日志
- 增加GPU内存使用监控
- 提供详细的错误信息和堆栈跟踪

### 2. 核心模型初始化日志 (`indextts/infer_v2.py`)
增加了以下组件的详细日志记录：

#### Qwen情感模型初始化
- 模型路径日志
- 成功/失败状态
- 详细错误信息

#### GPT模型初始化  
- UnifiedVoice实例创建
- 模型权重加载
- 设备迁移
- FP16/FP32模式设置
- 完整的错误追踪

#### SeamlessM4T特征提取器
- 模型下载和初始化状态
- 错误处理

#### 语义模型初始化
- W2V统计文件路径
- 设备迁移状态
- 详细错误信息

#### 语义编码器初始化
- 权重下载进度
- 模型加载状态
- 错误处理

#### S2Mel模型初始化
- 模型实例创建
- 权重加载
- 设备迁移
- 缓存设置
- 完整状态跟踪

#### CAMPPlus模型初始化
- 模型文件下载路径
- 加载状态
- 错误处理

#### **BigVGAN模型初始化（重点增强）**
- BigVGAN模型名称显示
- CUDA内核使用状态
- `from_pretrained`调用状态
- 模型实例创建状态  
- 设备迁移状态
- 权重标准化移除状态
- 评估模式设置状态
- **详细的错误类型和堆栈跟踪**

## 诊断工具

### 1. 模型初始化测试脚本 (`test_model_init.py`)
- 独立测试IndexTTS2初始化
- CUDA设备检测
- 详细的错误报告

### 2. BigVGAN下载测试脚本 (`test_bigvgan_download.py`)  
- 单独测试BigVGAN模型下载
- 测试不同CUDA内核选项
- 下载时间统计

## 使用方法

### 运行环境检查工具
```bash
uv run check_environment.py
```

### 运行带详细日志的API服务器（安全模式）
```bash
# 方法1：使用启动脚本（推荐）
./start_api_safe.sh

# 方法2：手动启动并禁用CUDA内核
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2 --disable-cuda-kernel

# 方法3：使用环境变量
export DISABLE_CUDA_KERNEL=1
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

### 运行带CUDA内核的API服务器（高性能模式）
```bash
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

## 日志输出示例

现在你应该能看到类似以下的详细日志：

```
>> 开始加载BigVGAN模型...
>> BigVGAN模型名称: nvidia/bigvgan_v2_22khz_80band_256x
>> 使用CUDA内核: True
>> 调用BigVGAN.from_pretrained...
>> ✓ BigVGAN模型实例创建成功
>> 将BigVGAN模型移动到设备: cuda:0
>> ✓ BigVGAN模型设备迁移成功
>> 移除BigVGAN权重标准化...
>> ✓ BigVGAN权重标准化移除成功
>> 设置BigVGAN为评估模式...
>> ✓ BigVGAN模型加载完全成功: nvidia/bigvgan_v2_22khz_80band_256x
```

如果出现错误，你会看到：
```
>> ✗ BigVGAN模型加载失败: [具体错误信息]
>> BigVGAN错误类型: [错误类型]
>> BigVGAN错误详情: [完整堆栈跟踪]
```

## CUDA内核编译问题解决方案

### 问题症状
- 服务器在初始化过程中卡住不动
- 日志显示CUDA内核编译错误
- GCC版本相关的编译错误

### 解决方案

#### 方案1：使用安全启动脚本（推荐）
```bash
./start_api_safe.sh
```

#### 方案2：禁用CUDA内核
```bash
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2 --disable-cuda-kernel
```

#### 方案3：设置环境变量
```bash
export DISABLE_CUDA_KERNEL=1
export TORCH_CUDA_ARCH_LIST=""
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

### 性能说明
- 禁用CUDA内核不会影响模型的核心功能
- 语音合成质量保持完全一致
- 推理速度可能有轻微下降（通常可忽略不计）
- GPU仍然会被正常使用，只是BigVGAN的某些优化被禁用

## 下一步调试

根据详细的日志输出，你可以确定问题的具体原因：

1. **CUDA编译问题**: 使用 `--disable-cuda-kernel` 参数
2. **网络问题**: 如果错误发生在模型下载阶段  
3. **内存问题**: 如果错误发生在设备迁移阶段
4. **模型文件问题**: 如果错误发生在权重加载阶段

这些详细的日志将帮助你精确定位问题所在。 