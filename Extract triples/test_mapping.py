#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 测试脚本，用于验证修改后的映射功能

from mapping2wiki import WikidataEntityMapper

def main():
    """测试主函数"""
    # 测试文件路径
    input_file = "output/entity_test.txt"
    output_file = "output/entity2id_test.txt"
    
    # 创建映射器并处理测试文件
    mapper = WikidataEntityMapper()
    mapper.process_entities(input_file, output_file)
    
    print("测试完成！请查看 output/entity2id_test.txt 文件")

if __name__ == "__main__":
    main()