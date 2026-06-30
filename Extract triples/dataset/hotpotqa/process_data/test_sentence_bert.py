"""
测试本地 sentence-bert 模型是否可以正常加载
"""
import os
import sys

def test_local_model():
    print("="*60)
    print("测试本地 sentence-bert 模型")
    print("="*60)
    
    # 1. 检查模型路径是否存在
    model_path = '../../../../sentence-bert'
    abs_model_path = os.path.abspath(model_path)
    
    print(f"\n[测试 1] 检查模型路径")
    print(f"  相对路径: {model_path}")
    print(f"  绝对路径: {abs_model_path}")
    
    if os.path.exists(model_path):
        print(f"  ✓ 模型路径存在")
    else:
        print(f"  ✗ 模型路径不存在")
        print(f"\n请确认 sentence-bert 模型在正确的位置。")
        return False
    
    # 2. 检查必要的模型文件
    print(f"\n[测试 2] 检查模型文件")
    required_files = [
        'config.json',
        'pytorch_model.bin',
        'vocab.txt',
        'tokenizer_config.json',
        'modules.json'
    ]
    
    all_files_exist = True
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 1024*1024:  # > 1MB
                size_str = f"{file_size / (1024*1024):.1f} MB"
            elif file_size > 1024:  # > 1KB
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} B"
            print(f"  ✓ {file} ({size_str})")
        else:
            print(f"  ✗ {file} 不存在")
            all_files_exist = False
    
    if not all_files_exist:
        print(f"\n模型文件不完整，请检查模型目录。")
        return False
    
    # 3. 尝试加载模型
    print(f"\n[测试 3] 加载 SentenceTransformer 模型")
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  正在加载模型...")
        model = SentenceTransformer(model_path)
        print(f"  ✓ 模型加载成功")
    except ImportError:
        print(f"  ✗ sentence-transformers 未安装")
        print(f"\n请安装: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return False
    
    # 4. 测试编码功能
    print(f"\n[测试 4] 测试编码功能")
    try:
        test_texts = ["Hello world", "测试文本"]
        embeddings = model.encode(test_texts, convert_to_numpy=True)
        print(f"  ✓ 编码成功")
        print(f"  输入文本数: {len(test_texts)}")
        print(f"  嵌入形状: {embeddings.shape}")
        print(f"  嵌入维度: {embeddings.shape[1]}")
    except Exception as e:
        print(f"  ✗ 编码失败: {e}")
        return False
    
    # 总结
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("✓ 本地 sentence-bert 模型可以正常使用！")
    print(f"\n模型信息:")
    print(f"  路径: {abs_model_path}")
    print(f"  嵌入维度: {embeddings.shape[1]}")
    print(f"\n可以开始生成混合嵌入了:")
    print(f"  python generate_hybrid_embeddings.py")
    
    return True

if __name__ == '__main__':
    success = test_local_model()
    sys.exit(0 if success else 1)

