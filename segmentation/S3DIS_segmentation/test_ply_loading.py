#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试PLY文件加载和标签提取
"""

import sys
import os
sys.path.append('../')

def test_ply_loading():
    try:
        from plyfile import PlyData
        print("plyfile库可用")
    except ImportError:
        print("正在安装plyfile库...")
        import subprocess
        subprocess.check_call(["pip", "install", "plyfile"])
        from plyfile import PlyData
        print("plyfile库安装完成")
    
    # 测试加载PLY文件
    ply_file = "../data/Toronto_3D/L001.ply"
    
    if not os.path.exists(ply_file):
        print(f"文件不存在: {ply_file}")
        return
    
    print(f"加载文件: {ply_file}")
    
    try:
        # 使用plyfile读取
        plydata = PlyData.read(ply_file)
        vertex_data = plydata['vertex']
        
        print(f"顶点数量: {len(vertex_data)}")
        print(f"数据类型: {type(vertex_data)}")
        
        # 正确访问字段名称
        if hasattr(vertex_data, 'dtype') and hasattr(vertex_data.dtype, 'names'):
            field_names = vertex_data.dtype.names
            print(f"可用字段: {field_names}")
            
            # 检查标签字段
            if 'scalar_Label' in field_names:
                labels = vertex_data['scalar_Label']
                print(f"标签字段类型: {labels.dtype}")
                print(f"标签范围: {labels.min()} 到 {labels.max()}")
                
                import numpy as np
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"唯一标签: {unique_labels}")
                print(f"标签分布: {dict(zip(unique_labels, counts))}")
                
                # 检查是否有非零标签
                non_zero_labels = labels[labels != 0]
                if len(non_zero_labels) > 0:
                    print(f"非零标签数量: {len(non_zero_labels)}")
                    print(f"非零标签示例: {non_zero_labels[:10]}")
                else:
                    print("警告: 所有标签都是0，可能需要检查数据")
            else:
                print("未找到scalar_Label字段")
        else:
            print("无法访问字段信息，尝试其他方法...")
            # 尝试直接访问数据
            try:
                # 列出所有可用的属性
                print(f"vertex_data属性: {dir(vertex_data)}")
                
                # 尝试访问一些常见字段
                if hasattr(vertex_data, 'x'):
                    print("找到x字段")
                if hasattr(vertex_data, 'scalar_Label'):
                    print("找到scalar_Label字段")
                    labels = vertex_data['scalar_Label']
                    print(f"标签范围: {labels.min()} 到 {labels.max()}")
            except Exception as e2:
                print(f"进一步检查失败: {e2}")
            
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ply_loading() 