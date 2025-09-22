#!/usr/bin/env python3
"""
IndexTTS2 模型预下载脚本
提前下载所有需要的外部模型，避免运行时网络卡顿
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_seamlessm4t():
    """下载SeamlessM4T特征提取器"""
    logger.info("开始下载SeamlessM4T特征提取器...")
    try:
        from transformers import SeamlessM4TFeatureExtractor
        start_time = time.time()
        
        # 下载模型
        extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        
        elapsed = time.time() - start_time
        logger.info(f"✓ SeamlessM4T下载成功! 耗时: {elapsed:.1f}秒")
        return True
        
    except Exception as e:
        logger.error(f"✗ SeamlessM4T下载失败: {e}")
        return False

def download_maskgct_semantic_codec():
    """下载MaskGCT语义编解码器"""
    logger.info("开始下载MaskGCT语义编解码器...")
    try:
        from huggingface_hub import hf_hub_download
        start_time = time.time()
        
        # 下载语义编解码器
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", 
            filename="semantic_codec/model.safetensors",
            cache_dir=None  # 使用默认缓存目录
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ MaskGCT语义编解码器下载成功! 耗时: {elapsed:.1f}秒")
        logger.info(f"下载位置: {semantic_code_ckpt}")
        return True
        
    except Exception as e:
        logger.error(f"✗ MaskGCT语义编解码器下载失败: {e}")
        return False

def download_campplus():
    """下载CAMPPlus模型"""
    logger.info("开始下载CAMPPlus模型...")
    try:
        from huggingface_hub import hf_hub_download
        start_time = time.time()
        
        # 下载CAMPPlus模型
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", 
            filename="campplus_cn_common.bin",
            cache_dir=None  # 使用默认缓存目录
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ CAMPPlus下载成功! 耗时: {elapsed:.1f}秒")
        logger.info(f"下载位置: {campplus_ckpt_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ CAMPPlus下载失败: {e}")
        return False

def download_bigvgan():
    """下载BigVGAN模型"""
    logger.info("开始下载BigVGAN模型...")
    try:
        # BigVGAN模型名称需要从配置文件中读取
        from omegaconf import OmegaConf
        
        config_path = "models/IndexTTS-2/config.yaml"
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            bigvgan_name = cfg.vocoder.name
            logger.info(f"BigVGAN模型名称: {bigvgan_name}")
            
            from indextts.s2mel.modules.bigvgan import bigvgan
            start_time = time.time()
            
            # 下载BigVGAN
            model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ BigVGAN下载成功! 耗时: {elapsed:.1f}秒")
            return True
        else:
            logger.error("✗ 配置文件不存在，无法获取BigVGAN模型名称")
            return False
            
    except Exception as e:
        logger.error(f"✗ BigVGAN下载失败: {e}")
        return False

def check_downloads():
    """检查已下载的模型"""
    logger.info("检查已下载的模型...")
    
    # 检查HuggingFace缓存目录
    from transformers.utils import default_cache_path
    cache_dir = default_cache_path
    logger.info(f"HuggingFace缓存目录: {cache_dir}")
    
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / 1024**3
        logger.info(f"缓存目录大小: {cache_size:.2f} GB")
    else:
        logger.info("缓存目录不存在")

def main():
    """主函数"""
    logger.info("开始预下载IndexTTS2所需的外部模型...")
    
    # 检查网络连接
    logger.info("检查网络连接...")
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        logger.info("✓ 网络连接正常")
    except Exception as e:
        logger.error(f"✗ 网络连接失败: {e}")
        logger.error("请检查网络连接或设置代理")
        return False
    
    # 设置环境变量（如果需要）
    hf_endpoint = os.environ.get('HF_ENDPOINT')
    if hf_endpoint:
        logger.info(f"使用HuggingFace镜像: {hf_endpoint}")
    
    # 检查当前下载状态
    check_downloads()
    
    # 定义下载任务
    download_tasks = [
        ("SeamlessM4T", download_seamlessm4t),
        ("MaskGCT语义编解码器", download_maskgct_semantic_codec),
        ("CAMPPlus", download_campplus),
        ("BigVGAN", download_bigvgan)
    ]
    
    # 串行下载（避免并发导致的网络问题）
    success_count = 0
    total_count = len(download_tasks)
    
    for task_name, task_func in download_tasks:
        logger.info(f"\n开始下载任务: {task_name}")
        try:
            if task_func():
                success_count += 1
            else:
                logger.error(f"任务失败: {task_name}")
        except Exception as e:
            logger.error(f"任务异常: {task_name} - {e}")
    
    # 最终检查
    logger.info(f"\n=== 下载完成 ===")
    logger.info(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        logger.info("✓ 所有模型下载成功!")
        logger.info("现在可以启动IndexTTS2 API服务器了")
        
        # 再次检查缓存
        check_downloads()
        
        return True
    else:
        logger.error("✗ 部分模型下载失败")
        logger.error("建议:")
        logger.error("1. 检查网络连接")
        logger.error("2. 设置代理: export https_proxy=http://proxy:port")
        logger.error("3. 使用镜像: export HF_ENDPOINT=https://hf-mirror.com")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 