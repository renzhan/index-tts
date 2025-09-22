#!/usr/bin/env python3
"""
快速检查IndexTTS2部署环境的脚本
用于快速诊断GPU占用5G后卡住的问题
"""

import os
import sys
import time
import logging
import subprocess

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_command(cmd, description):
    """检查命令是否可用"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✓ {description}")
            return True
        else:
            logger.error(f"✗ {description} - {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} - 命令超时")
        return False
    except Exception as e:
        logger.error(f"✗ {description} - {e}")
        return False

def main():
    """快速检查"""
    logger.info("=== IndexTTS2 快速环境检查 ===\n")
    
    issues = []
    
    # 1. 检查GPU状态
    logger.info("1. 检查GPU状态...")
    if not check_command("nvidia-smi", "nvidia-smi可用"):
        issues.append("GPU驱动问题")
    
    # 2. 检查Python环境
    logger.info("\n2. 检查Python环境...")
    try:
        import torch
        logger.info(f"✓ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✓ GPU设备: {device_name}")
            logger.info(f"✓ GPU总内存: {total_memory:.1f} GB")
            
            if total_memory < 8.0:
                issues.append(f"GPU内存不足 ({total_memory:.1f} GB < 8 GB)")
                logger.warning(f"⚠ GPU内存可能不足: {total_memory:.1f} GB")
        else:
            issues.append("CUDA不可用")
            logger.error("✗ CUDA不可用")
            
    except ImportError:
        issues.append("PyTorch未安装")
        logger.error("✗ PyTorch未安装")
    
    # 3. 检查网络连接
    logger.info("\n3. 检查网络连接...")
    if not check_command("ping -c 1 8.8.8.8", "网络连接"):
        issues.append("网络连接问题")
    
    if not check_command("curl -s --connect-timeout 5 https://huggingface.co", "HuggingFace连接"):
        issues.append("HuggingFace无法访问")
        logger.warning("⚠ 建议使用离线模式或设置代理")
    
    # 4. 检查模型文件
    logger.info("\n4. 检查模型文件...")
    model_dir = "models/IndexTTS-2"
    required_files = [
        "config.yaml", "gpt.pth", "s2mel.pth", "bpe.model",
        "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024**2
            logger.info(f"✓ {file} ({size:.1f} MB)")
        else:
            missing_files.append(file)
            logger.error(f"✗ 缺少文件: {file}")
    
    if missing_files:
        issues.append(f"缺少模型文件: {', '.join(missing_files)}")
    
    # 5. 检查依赖库
    logger.info("\n5. 检查关键依赖库...")
    dependencies = [
        ("transformers", "Transformers库"),
        ("omegaconf", "OmegaConf配置库"),
        ("torchaudio", "TorchAudio库"),
        ("fastapi", "FastAPI库"),
        ("psutil", "系统监控库")
    ]
    
    for module, desc in dependencies:
        try:
            __import__(module)
            logger.info(f"✓ {desc}")
        except ImportError:
            issues.append(f"缺少依赖: {module}")
            logger.error(f"✗ 缺少{desc}")
    
    # 6. 检查环境变量
    logger.info("\n6. 检查环境变量...")
    env_vars = [
        ("CUDA_VISIBLE_DEVICES", "GPU设备选择"),
        ("PYTORCH_CUDA_ALLOC_CONF", "CUDA内存分配配置"),
    ]
    
    for var, desc in env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"✓ {var}={value}")
        else:
            logger.info(f"- {var} 未设置")
    
    # 输出问题总结
    logger.info("\n=== 检查总结 ===")
    if not issues:
        logger.info("✓ 环境检查通过！")
        logger.info("\n建议下一步:")
        logger.info("1. 运行 'python debug_model_loading.py' 进行详细诊断")
        logger.info("2. 使用更新后的api_server.py启动服务")
    else:
        logger.error("✗ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        
        logger.info("\n建议解决方案:")
        if "GPU内存不足" in str(issues):
            logger.info("- 考虑使用CPU模式: device='cpu'")
            logger.info("- 或升级GPU硬件")
        
        if "网络连接问题" in str(issues) or "HuggingFace无法访问" in str(issues):
            logger.info("- 设置代理: export https_proxy=http://proxy:port")
            logger.info("- 或使用离线模式部署")
        
        if "缺少模型文件" in str(issues):
            logger.info("- 重新下载完整的模型文件")
            logger.info("- 检查下载过程是否被中断")
        
        if "缺少依赖" in str(issues):
            logger.info("- 安装缺少的依赖库")
            logger.info("- 重新创建虚拟环境")

if __name__ == "__main__":
    main() 