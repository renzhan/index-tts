#!/usr/bin/env python3
"""
调试IndexTTS2模型加载问题的脚本
分步骤测试各个组件的加载情况
"""

import os
import sys
import time
import logging
import signal
import threading
from contextlib import contextmanager

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import torch
from omegaconf import OmegaConf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """超时上下文管理器"""
    def signal_handler(signum, frame):
        raise TimeoutException(f"操作超时 ({seconds}秒)")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def log_gpu_memory(step_name):
    """记录GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        logger.info(f"[{step_name}] GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB, 峰值: {max_allocated:.2f}GB")
    else:
        logger.info(f"[{step_name}] 使用CPU模式")

def test_basic_imports():
    """测试基本导入"""
    logger.info("=== 测试基本导入 ===")
    try:
        # 测试PyTorch相关导入
        logger.info("导入torch相关模块...")
        import torch
        import torchaudio
        logger.info(f"✓ PyTorch版本: {torch.__version__}")
        
        # 测试Transformers导入
        logger.info("导入transformers...")
        from transformers import GPT2PreTrainedModel
        logger.info("✓ Transformers导入成功")
        
        # 测试OmegaConf
        logger.info("测试OmegaConf...")
        from omegaconf import OmegaConf
        logger.info("✓ OmegaConf导入成功")
        
        return True
    except Exception as e:
        logger.error(f"✗ 基本导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    logger.info("=== 测试配置文件加载 ===")
    try:
        model_dir = "models/IndexTTS-2"
        config_path = os.path.join(model_dir, "config.yaml")
        
        if not os.path.exists(config_path):
            logger.error(f"✗ 配置文件不存在: {config_path}")
            return False
        
        logger.info(f"加载配置文件: {config_path}")
        cfg = OmegaConf.load(config_path)
        logger.info("✓ 配置文件加载成功")
        logger.info(f"配置版本: {getattr(cfg, 'version', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"✗ 配置文件加载失败: {e}")
        return False

def test_model_files():
    """测试模型文件完整性"""
    logger.info("=== 测试模型文件完整性 ===")
    model_dir = "models/IndexTTS-2"
    
    required_files = [
        "config.yaml", "gpt.pth", "s2mel.pth", "bpe.model",
        "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"✗ 缺少文件: {file}")
            return False
        
        file_size = os.path.getsize(file_path) / 1024**2
        logger.info(f"✓ {file} ({file_size:.1f} MB)")
    
    return True

def test_device_initialization():
    """测试设备初始化"""
    logger.info("=== 测试设备初始化 ===")
    
    # 测试CUDA
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA可用: {torch.version.cuda}")
        logger.info(f"✓ GPU设备: {torch.cuda.get_device_name(0)}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✓ GPU总内存: {total_memory:.2f} GB")
        
        # 测试简单的GPU操作
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            logger.info("✓ GPU基本操作测试通过")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"✗ GPU操作测试失败: {e}")
            return False
    else:
        logger.warning("⚠ CUDA不可用，将使用CPU模式")
    
    return True

def test_step_by_step_loading():
    """分步骤测试模型加载"""
    logger.info("=== 分步骤测试模型加载 ===")
    
    model_dir = "models/IndexTTS-2"
    config_path = os.path.join(model_dir, "config.yaml")
    
    try:
        # 步骤1: 加载配置
        logger.info("步骤1: 加载配置文件...")
        log_gpu_memory("加载配置前")
        cfg = OmegaConf.load(config_path)
        log_gpu_memory("加载配置后")
        
        # 步骤2: 设备检测
        logger.info("步骤2: 设备检测...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_fp16 = torch.cuda.is_available()
        logger.info(f"设备: {device}, FP16: {use_fp16}")
        
        # 步骤3: 测试GPT模型加载
        logger.info("步骤3: 测试GPT模型加载...")
        log_gpu_memory("GPT加载前")
        
        try:
            with timeout(120):  # 2分钟超时
                from indextts.gpt.model_v2 import UnifiedVoice
                from indextts.utils import load_checkpoint
                
                logger.info("创建GPT模型实例...")
                gpt = UnifiedVoice(**cfg.gpt)
                
                logger.info("加载GPT权重...")
                gpt_path = os.path.join(model_dir, cfg.gpt_checkpoint)
                load_checkpoint(gpt, gpt_path)
                
                logger.info("移动GPT到设备...")
                gpt = gpt.to(device)
                
                if use_fp16:
                    logger.info("转换GPT到半精度...")
                    gpt.eval().half()
                else:
                    gpt.eval()
                
                logger.info("✓ GPT模型加载成功")
                log_gpu_memory("GPT加载后")
                
        except TimeoutException:
            logger.error("✗ GPT模型加载超时")
            return False
        except Exception as e:
            logger.error(f"✗ GPT模型加载失败: {e}")
            return False
        
        # 步骤4: 测试语义模型加载
        logger.info("步骤4: 测试语义模型加载...")
        log_gpu_memory("语义模型加载前")
        
        try:
            with timeout(120):  # 2分钟超时
                logger.info("加载SeamlessM4T特征提取器...")
                from transformers import SeamlessM4TFeatureExtractor
                extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
                logger.info("✓ SeamlessM4T特征提取器加载成功")
                
                logger.info("加载语义模型...")
                from indextts.semantic_model import build_semantic_model
                semantic_model, semantic_mean, semantic_std = build_semantic_model(
                    os.path.join(model_dir, cfg.w2v_stat))
                semantic_model = semantic_model.to(device)
                semantic_model.eval()
                
                logger.info("✓ 语义模型加载成功")
                log_gpu_memory("语义模型加载后")
                
        except TimeoutException:
            logger.error("✗ 语义模型加载超时")
            return False
        except Exception as e:
            logger.error(f"✗ 语义模型加载失败: {e}")
            return False
        
        logger.info("✓ 所有步骤测试完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ 分步骤测试失败: {e}")
        import traceback
        logger.error(f"详细错误:\n{traceback.format_exc()}")
        return False

def main():
    """主函数"""
    logger.info("开始IndexTTS2加载问题诊断...")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("清理GPU缓存")
    
    tests = [
        ("基本导入", test_basic_imports),
        ("配置文件加载", test_config_loading),
        ("模型文件检查", test_model_files),
        ("设备初始化", test_device_initialization),
        ("分步骤加载", test_step_by_step_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n开始测试: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            elapsed = time.time() - start_time
            results[test_name] = result
            
            if result:
                logger.info(f"✓ {test_name} 通过 (耗时: {elapsed:.2f}秒)")
            else:
                logger.error(f"✗ {test_name} 失败 (耗时: {elapsed:.2f}秒)")
                break  # 如果测试失败，停止后续测试
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"✗ {test_name} 异常: {e} (耗时: {elapsed:.2f}秒)")
            results[test_name] = False
            break
    
    # 输出总结
    logger.info("\n=== 测试总结 ===")
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    # 最终建议
    if all(results.values()):
        logger.info("\n✓ 所有测试通过！模型应该可以正常加载。")
        logger.info("如果API服务器仍然卡住，可能是网络下载问题或其他依赖项问题。")
    else:
        logger.info("\n✗ 发现问题！请根据上述失败的测试进行排查。")
    
    return all(results.values())

if __name__ == "__main__":
    main() 