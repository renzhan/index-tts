#!/usr/bin/env python3
"""
测试IndexTTS2关键导入的脚本
验证load_checkpoint导入问题是否已修复
"""

import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

def test_critical_imports():
    """测试关键导入"""
    
    print("=== 测试IndexTTS2关键导入 ===")
    
    try:
        print("1. 测试 load_checkpoint 导入...")
        from indextts.utils.checkpoint import load_checkpoint
        print("✓ indextts.utils.checkpoint.load_checkpoint 导入成功")
        
        # 也测试从包级别导入
        from indextts.utils import load_checkpoint as lc2
        print("✓ indextts.utils.load_checkpoint 导入成功")
        
    except ImportError as e:
        print(f"✗ load_checkpoint 导入失败: {e}")
        return False
    
    try:
        print("\n2. 测试 GPT 模型导入...")
        from indextts.gpt.model_v2 import UnifiedVoice
        print("✓ UnifiedVoice 导入成功")
        
    except ImportError as e:
        print(f"✗ UnifiedVoice 导入失败: {e}")
        return False
    
    try:
        print("\n3. 测试 MaskGCT 工具导入...")
        from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
        print("✓ MaskGCT 工具导入成功")
        
    except ImportError as e:
        print(f"✗ MaskGCT 工具导入失败: {e}")
        return False
    
    try:
        print("\n4. 测试文本处理导入...")
        from indextts.utils.front import TextNormalizer, TextTokenizer
        print("✓ 文本处理工具导入成功")
        
    except ImportError as e:
        print(f"✗ 文本处理工具导入失败: {e}")
        return False
    
    try:
        print("\n5. 测试 S2Mel 模型导入...")
        from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
        print("✓ S2Mel 模型导入成功")
        
    except ImportError as e:
        print(f"✗ S2Mel 模型导入失败: {e}")
        return False
    
    try:
        print("\n6. 测试 BigVGAN 导入...")
        from indextts.s2mel.modules.bigvgan import bigvgan
        print("✓ BigVGAN 导入成功")
        
    except ImportError as e:
        print(f"✗ BigVGAN 导入失败: {e}")
        return False
    
    try:
        print("\n7. 测试 CAMPPlus 导入...")
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        print("✓ CAMPPlus 导入成功")
        
    except ImportError as e:
        print(f"✗ CAMPPlus 导入失败: {e}")
        return False
    
    try:
        print("\n8. 测试 Transformers 依赖...")
        from transformers import SeamlessM4TFeatureExtractor
        print("✓ SeamlessM4TFeatureExtractor 导入成功")
        
    except ImportError as e:
        print(f"✗ SeamlessM4TFeatureExtractor 导入失败: {e}")
        return False
    
    print("\n9. 最终测试: 尝试导入 IndexTTS2...")
    try:
        from indextts.infer_v2 import IndexTTS2
        print("✓ IndexTTS2 类导入成功！")
        return True
        
    except ImportError as e:
        print(f"✗ IndexTTS2 导入失败: {e}")
        return False

def main():
    """主函数"""
    success = test_critical_imports()
    
    print(f"\n=== 测试结果 ===")
    if success:
        print("✓ 所有关键导入测试通过！")
        print("load_checkpoint 导入问题已修复")
        print("现在可以尝试重新运行:")
        print("  python debug_model_loading.py")
        print("  或直接启动 API 服务器")
    else:
        print("✗ 仍有导入问题需要解决")
    
    return success

if __name__ == "__main__":
    main() 