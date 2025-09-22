# VS Code 开发指南

## 配置说明

已为IndexTTS2项目创建了完整的VS Code开发环境配置：

### 📁 配置文件

- **`.vscode/launch.json`** - 调试配置
- **`.vscode/tasks.json`** - 任务快捷方式
- **`.vscode/settings.json`** - 编辑器设置

## 🚀 调试配置 (launch.json)

### 主要调试配置

1. **IndexTTS2 API Server** - 标准GPU模式
   - 命令: `uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2`
   - 包含GPU优化环境变量
   - 使用HuggingFace镜像源

2. **IndexTTS2 API Server (CPU Mode)** - CPU模式
   - 禁用GPU (`CUDA_VISIBLE_DEVICES=""`)
   - 适用于GPU内存不足的情况

3. **IndexTTS2 API Server (Debug Mode)** - 开发调试模式
   - 启用auto-reload
   - 仅监听本地连接 (127.0.0.1)
   - 详细调试信息

### 诊断工具配置

4. **Test Model Loading** - 模型加载测试
5. **Quick Environment Check** - 环境快速检查
6. **Network Debug** - 网络连接诊断
7. **Predownload Models** - 模型预下载

## ⚡ 任务配置 (tasks.json)

### 快捷任务

- **`Ctrl+Shift+P` → `Tasks: Run Task`** 选择任务:

1. **Start IndexTTS2 API Server** (默认构建任务)
   - 快捷键: `Ctrl+Shift+B`
   
2. **Start API Server (with Mirror)** - 使用镜像源启动
3. **Quick Environment Check** - 环境检查
4. **Test Model Loading** - 模型加载测试
5. **Network Debug** - 网络诊断
6. **Predownload Models** - 预下载模型
7. **Test Imports** - 导入测试
8. **Clean Cache** - 清理缓存
9. **Setup Network Solution** - 网络问题解决方案

## 🛠️ 使用方法

### 1. 启动API服务器

#### 方法A: 使用调试模式
1. 按 `F5` 或点击"运行和调试"
2. 选择 "IndexTTS2 API Server"
3. 点击绿色播放按钮

#### 方法B: 使用任务
1. 按 `Ctrl+Shift+B`
2. 选择 "Start IndexTTS2 API Server"

#### 方法C: 使用命令面板
1. 按 `Ctrl+Shift+P`
2. 输入 "Tasks: Run Task"
3. 选择相应任务

### 2. 调试代码

#### 设置断点
- 在代码行号左侧点击设置断点
- 或在代码行按 `F9`

#### 调试控制
- `F5` - 继续执行
- `F10` - 单步跳过
- `F11` - 单步进入
- `Shift+F11` - 单步跳出
- `Shift+F5` - 停止调试

### 3. 环境诊断

#### 快速检查
```
Ctrl+Shift+P → Tasks: Run Task → Quick Environment Check
```

#### 网络问题诊断
```
Ctrl+Shift+P → Tasks: Run Task → Network Debug
```

#### 模型加载测试
```
Ctrl+Shift+P → Tasks: Run Task → Test Model Loading
```

### 4. 预下载模型
```
Ctrl+Shift+P → Tasks: Run Task → Predownload Models
```

## 🔧 环境变量

配置文件中已设置的重要环境变量：

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=1
HF_HOME=./hf_cache
TRANSFORMERS_CACHE=./hf_cache/transformers
HF_ENDPOINT=https://hf-mirror.com
```

## 📋 推荐的工作流程

### 首次启动
1. 运行 "Quick Environment Check"
2. 如有网络问题，运行 "Network Debug"
3. 预下载模型: "Predownload Models"
4. 启动API服务器: `F5` 或 `Ctrl+Shift+B`

### 开发调试
1. 设置断点
2. 选择 "IndexTTS2 API Server (Debug Mode)"
3. 按 `F5` 开始调试
4. 修改代码后自动重载

### 问题排查
1. 使用 "Test Model Loading" 检查模型加载
2. 使用 "Network Debug" 检查网络连接
3. 查看集成终端中的详细日志

## 🎯 快捷键汇总

| 操作 | 快捷键 |
|------|---------|
| 启动调试 | `F5` |
| 运行构建任务 | `Ctrl+Shift+B` |
| 打开命令面板 | `Ctrl+Shift+P` |
| 设置断点 | `F9` |
| 单步调试 | `F10` |
| 停止调试 | `Shift+F5` |
| 打开终端 | `Ctrl+Shift+` |

## 🔍 常见问题

### Q: 调试时找不到模块
A: 检查 `settings.json` 中的 `python.analysis.extraPaths` 配置

### Q: 环境变量不生效
A: 确保在VS Code中重新加载窗口 (`Ctrl+Shift+P` → "Developer: Reload Window")

### Q: 网络下载卡住
A: 运行 "Setup Network Solution" 任务获取解决方案

### Q: GPU内存不足
A: 使用 "IndexTTS2 API Server (CPU Mode)" 调试配置

## 📦 推荐插件

配置中已包含推荐插件列表：

- Python
- Pylance
- Black Formatter
- Flake8
- Jupyter
- JSON
- YAML
- CMake Tools

VS Code会自动提示安装这些插件。

现在你可以在VS Code中高效地开发和调试IndexTTS2项目了！ 