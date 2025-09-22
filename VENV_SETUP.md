# 虚拟环境配置说明

## VS Code配置更新

已更新所有VS Code配置文件，确保使用 `.venv` 虚拟环境中的Python。

## 🔧 配置更改

### 1. launch.json - 调试配置
所有调试配置现在都指定使用虚拟环境的Python：

```json
"python": "${workspaceFolder}/.venv/bin/python"
```

### 2. tasks.json - 任务配置
所有Python任务现在都使用虚拟环境：

```json
"command": "${workspaceFolder}/.venv/bin/python"
```

### 3. settings.json - 编辑器设置
默认Python解释器路径更新为：

```json
"python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
```

## 🖥️ 跨平台兼容性

### Linux/Mac 系统
- Python路径: `${workspaceFolder}/.venv/bin/python`
- PYTHONPATH分隔符: `:`

### Windows 系统
- Python路径: `${workspaceFolder}/.venv/Scripts/python.exe`
- PYTHONPATH分隔符: `;`

如果你在Windows系统下使用，需要手动修改launch.json中的Python路径。

## ⚡ 使用方法

### 1. 确保虚拟环境已激活

#### Linux/Mac:
```bash
source .venv/bin/activate
```

#### Windows:
```bash
.venv\Scripts\activate
```

### 2. 验证Python路径

#### 在虚拟环境中运行:
```bash
which python  # Linux/Mac
where python  # Windows
```

应该显示 `.venv` 目录下的Python路径。

### 3. VS Code中选择解释器

1. 按 `Ctrl+Shift+P`
2. 输入 "Python: Select Interpreter"
3. 选择 `./.venv/bin/python` (或Windows下的 `./.venv/Scripts/python.exe`)

### 4. 验证配置

运行以下任务验证配置：
- `Ctrl+Shift+P` → `Tasks: Run Task` → `Quick Environment Check`

## 🚀 启动API服务器

现在所有方式都会使用虚拟环境的Python：

### 方法1: 调试模式
```
F5 → 选择 "IndexTTS2 API Server"
```

### 方法2: 任务模式
```
Ctrl+Shift+B → "Start IndexTTS2 API Server"
```

### 方法3: 命令行
```bash
# 激活虚拟环境
source .venv/bin/activate

# 启动服务器
uv run api_server.py --host 0.0.0.0 --port 8000 --model-dir models/IndexTTS-2
```

## 🔍 故障排查

### 问题1: 找不到模块
如果出现模块导入错误：
1. 确认虚拟环境已激活
2. 检查 `PYTHONPATH` 环境变量设置
3. 验证依赖包是否安装在虚拟环境中

### 问题2: Python解释器错误
如果VS Code使用了错误的Python：
1. 按 `Ctrl+Shift+P`
2. 选择 "Python: Select Interpreter"
3. 手动选择 `.venv` 目录下的Python

### 问题3: Windows路径问题
Windows用户需要确保使用正确的路径：
- Python可执行文件: `.venv/Scripts/python.exe`
- 使用反斜杠 `\` 作为路径分隔符

## 📋 验证清单

- [ ] 虚拟环境已创建并激活
- [ ] VS Code已选择正确的Python解释器
- [ ] 调试配置使用虚拟环境Python
- [ ] 任务配置使用虚拟环境Python
- [ ] 环境变量设置正确
- [ ] 可以成功运行诊断脚本

## 🎯 推荐工作流程

1. **创建虚拟环境** (如果还没有):
   ```bash
   python -m venv .venv
   ```

2. **激活虚拟环境**:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

4. **在VS Code中选择解释器**:
   `Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `.venv` 下的Python

5. **运行环境检查**:
   `Ctrl+Shift+P` → `Tasks: Run Task` → `Quick Environment Check`

6. **启动API服务器**:
   按 `F5` 或 `Ctrl+Shift+B`

现在所有的VS Code配置都已正确设置为使用 `.venv` 虚拟环境中的Python！ 