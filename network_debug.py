#!/usr/bin/env python3
"""
网络连接诊断脚本
检测IndexTTS2需要下载的模型和网络连接状态
"""

import os
import sys
import time
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def test_network_connection():
    """测试网络连接"""
    print("=== 网络连接测试 ===")
    
    urls_to_test = [
        ("Google DNS", "https://8.8.8.8"),
        ("百度", "https://www.baidu.com"),
        ("HuggingFace", "https://huggingface.co"),
        ("GitHub", "https://github.com"),
        ("HuggingFace Hub", "https://huggingface.co/facebook/w2v-bert-2.0"),
        ("Amphion MaskGCT", "https://huggingface.co/amphion/MaskGCT"),
        ("FunASR CAMPPlus", "https://huggingface.co/funasr/campplus")
    ]
    
    def test_url(name, url):
        try:
            response = requests.get(url, timeout=10)
            return f"✓ {name}: 可访问 (状态码: {response.status_code})"
        except requests.exceptions.Timeout:
            return f"✗ {name}: 超时"
        except requests.exceptions.ConnectionError:
            return f"✗ {name}: 连接失败"
        except Exception as e:
            return f"✗ {name}: 错误 - {e}"
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(test_url, name, url) for name, url in urls_to_test]
        
        for future in futures:
            try:
                result = future.result(timeout=15)
                print(result)
            except FutureTimeoutError:
                print("✗ 测试超时")

def check_huggingface_cache():
    """检查HuggingFace缓存"""
    print("\n=== HuggingFace缓存检查 ===")
    
    # 检查默认缓存目录
    import transformers
    cache_dir = transformers.file_utils.default_cache_path
    print(f"默认缓存目录: {cache_dir}")
    
    # 检查环境变量设置的缓存目录
    hf_home = os.environ.get('HF_HOME', '未设置')
    transformers_cache = os.environ.get('TRANSFORMERS_CACHE', '未设置')
    
    print(f"HF_HOME: {hf_home}")
    print(f"TRANSFORMERS_CACHE: {transformers_cache}")
    
    # 检查是否有离线模式
    offline_mode = os.environ.get('TRANSFORMERS_OFFLINE', '0')
    print(f"离线模式: {'是' if offline_mode == '1' else '否'}")

def test_model_downloads():
    """测试模型下载"""
    print("\n=== 模型下载测试 ===")
    
    models_to_test = [
        "facebook/w2v-bert-2.0",
        "amphion/MaskGCT", 
        "funasr/campplus"
    ]
    
    for model_name in models_to_test:
        print(f"\n测试模型: {model_name}")
        try:
            # 测试能否获取模型信息
            from huggingface_hub import model_info
            info = model_info(model_name, timeout=10)
            print(f"✓ 模型信息获取成功: {info.modelId}")
        except Exception as e:
            print(f"✗ 模型信息获取失败: {e}")

def test_specific_downloads():
    """测试具体的下载任务"""
    print("\n=== 具体下载测试 ===")
    
    # 测试SeamlessM4T特征提取器
    print("1. 测试SeamlessM4T特征提取器下载...")
    try:
        from transformers import SeamlessM4TFeatureExtractor
        print("尝试下载 facebook/w2v-bert-2.0...")
        
        # 设置较短的超时时间来测试
        import transformers
        original_timeout = getattr(transformers.file_utils, 'TORCH_HUB_TIMEOUT', None)
        transformers.file_utils.TORCH_HUB_TIMEOUT = 30  # 30秒超时
        
        start_time = time.time()
        extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        elapsed = time.time() - start_time
        print(f"✓ SeamlessM4T下载成功 (耗时: {elapsed:.1f}秒)")
        
    except Exception as e:
        print(f"✗ SeamlessM4T下载失败: {e}")
    
    # 测试语义编解码器下载
    print("\n2. 测试MaskGCT语义编解码器下载...")
    try:
        from huggingface_hub import hf_hub_download
        print("尝试下载 amphion/MaskGCT 语义编解码器...")
        
        start_time = time.time()
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", 
            filename="semantic_codec/model.safetensors",
            timeout=30
        )
        elapsed = time.time() - start_time
        print(f"✓ MaskGCT语义编解码器下载成功 (耗时: {elapsed:.1f}秒)")
        print(f"下载位置: {semantic_code_ckpt}")
        
    except Exception as e:
        print(f"✗ MaskGCT语义编解码器下载失败: {e}")
    
    # 测试CAMPPlus下载
    print("\n3. 测试CAMPPlus模型下载...")
    try:
        from huggingface_hub import hf_hub_download
        print("尝试下载 funasr/campplus...")
        
        start_time = time.time()
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", 
            filename="campplus_cn_common.bin",
            timeout=30
        )
        elapsed = time.time() - start_time
        print(f"✓ CAMPPlus下载成功 (耗时: {elapsed:.1f}秒)")
        print(f"下载位置: {campplus_ckpt_path}")
        
    except Exception as e:
        print(f"✗ CAMPPlus下载失败: {e}")

def provide_solutions():
    """提供解决方案"""
    print("\n=== 解决方案建议 ===")
    
    print("如果网络连接有问题，可以尝试以下解决方案:")
    print()
    print("方案1: 设置代理")
    print("export https_proxy=http://your-proxy:port")
    print("export http_proxy=http://your-proxy:port")
    print()
    print("方案2: 使用镜像源")
    print("export HF_ENDPOINT=https://hf-mirror.com")
    print()
    print("方案3: 离线模式部署")
    print("1. 在有网络的机器上预下载模型:")
    print("   python -c \"from transformers import SeamlessM4TFeatureExtractor; SeamlessM4TFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')\"")
    print("2. 将缓存目录复制到目标服务器")
    print("3. 设置离线环境变量:")
    print("   export TRANSFORMERS_OFFLINE=1")
    print("   export HF_DATASETS_OFFLINE=1")
    print()
    print("方案4: 修改超时设置")
    print("可以在代码中增加超时时间或重试机制")

def main():
    print("IndexTTS2 网络连接诊断工具")
    print("=" * 50)
    
    # 基础网络测试
    test_network_connection()
    
    # HuggingFace缓存检查
    check_huggingface_cache()
    
    # 模型信息测试
    test_model_downloads()
    
    # 具体下载测试
    test_specific_downloads()
    
    # 提供解决方案
    provide_solutions()

if __name__ == "__main__":
    main() 