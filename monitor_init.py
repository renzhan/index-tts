#!/usr/bin/env python3
"""
IndexTTS2模型初始化监控脚本
用于诊断模型加载过程中的问题
"""

import os
import sys
import time
import psutil
import threading
import logging
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import torch
from indextts.infer_v2 import IndexTTS2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_init.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控系统资源"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("开始监控系统资源...")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("停止监控系统资源")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU和内存使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU信息
                gpu_info = ""
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    cached = torch.cuda.memory_reserved(0) / 1024**3
                    gpu_info = f" | GPU: {allocated:.2f}GB/{cached:.2f}GB"
                
                logger.info(f"系统状态 - CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%{gpu_info}")
                
                # 检查是否有进程卡死
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"监控过程出错: {e}")
                break

def hook_model_methods():
    """为IndexTTS2的关键方法添加钩子函数"""
    
    # 保存原始方法
    original_init = IndexTTS2.__init__
    
    def logged_init(self, *args, **kwargs):
        logger.info("=== IndexTTS2.__init__ 开始 ===")
        logger.info(f"参数: args={args}, kwargs={kwargs}")
        
        try:
            # 记录各个初始化步骤
            logger.info("步骤1: 设备和配置初始化...")
            
            # 调用原始初始化方法
            result = original_init(self, *args, **kwargs)
            
            logger.info("=== IndexTTS2.__init__ 完成 ===")
            return result
            
        except Exception as e:
            logger.error(f"=== IndexTTS2.__init__ 失败: {e} ===")
            raise
    
    # 替换方法
    IndexTTS2.__init__ = logged_init

def test_model_initialization():
    """测试模型初始化"""
    model_dir = "models/IndexTTS-2"
    config_path = os.path.join(model_dir, "config.yaml")
    
    # 检查文件
    logger.info("检查模型文件...")
    required_files = [
        "config.yaml", "gpt.pth", "s2mel.pth", "bpe.model",
        "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"缺少文件: {file_path}")
            return False
        file_size = os.path.getsize(file_path) / 1024**2
        logger.info(f"✓ {file} ({file_size:.1f} MB)")
    
    # 开始监控
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    try:
        # 添加方法钩子
        hook_model_methods()
        
        logger.info("开始初始化IndexTTS2模型...")
        start_time = time.time()
        
        # 初始化模型
        model = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=torch.cuda.is_available(),
            use_deepspeed=False,
            device=None
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"模型初始化成功! 耗时: {elapsed_time:.2f}秒")
        
        # 最终内存报告
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"最终GPU内存使用: {allocated:.2f} GB (已分配), {cached:.2f} GB (已缓存)")
        
        return True
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        import traceback
        logger.error(f"完整错误信息:\n{traceback.format_exc()}")
        return False
        
    finally:
        monitor.stop_monitoring()

def main():
    """主函数"""
    logger.info("开始IndexTTS2初始化诊断...")
    
    # 系统信息
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU总内存: {total_memory:.2f} GB")
    
    # 内存信息
    memory = psutil.virtual_memory()
    logger.info(f"系统总内存: {memory.total / 1024**3:.2f} GB")
    logger.info(f"可用内存: {memory.available / 1024**3:.2f} GB")
    
    # 运行测试
    success = test_model_initialization()
    
    if success:
        logger.info("✓ 模型初始化测试成功!")
    else:
        logger.error("✗ 模型初始化测试失败!")
    
    return success

if __name__ == "__main__":
    main() 