#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Toronto 3D标签提取
"""

import os
import numpy as np

def test_label_extraction():
    # 测试文件路径
    test_file = "../data/Toronto_3D/L001.ply"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    try:
        from plyfile import PlyData
        print(f"正在测试文件: {test_file}")
        
        # 读取PLY文件
        plydata = PlyData.read(test_file)
        vertex_data = plydata['vertex']
        
        print(f"顶点数量: {len(vertex_data):,}")
        print(f"数据类型: {type(vertex_data)}")
        
        # 正确访问plyfile数据
        try:
            # plyfile的正确访问方式
            actual_data = vertex_data.data
            field_names = actual_data.dtype.names
            print(f"字段列表: {list(field_names)}")
            
            # 检查scalar_Label字段
            if 'scalar_Label' in field_names:
                labels = actual_data['scalar_Label']
                print(f"\nscalar_Label字段信息:")
                print(f"  数据类型: {labels.dtype}")
                print(f"  数组形状: {labels.shape}")
                print(f"  值范围: {labels.min()} ~ {labels.max()}")
                
                # 详细分析标签分布
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"  唯一标签: {unique_labels}")
                print(f"  标签数量: {len(unique_labels)}")
                
                # 显示标签分布
                print(f"\n标签分布:")
                for label, count in zip(unique_labels, counts):
                    percentage = count / len(labels) * 100
                    print(f"  标签 {label}: {count:,} 个点 ({percentage:.2f}%)")
                
                # 检查前100个点的标签值作为样本
                print(f"\n前10个点的标签值: {labels[:10]}")
                
                # 如果所有标签都是0，可能需要特殊处理
                if len(unique_labels) == 1 and unique_labels[0] == 0:
                    print("\n⚠️ 警告: 所有标签都是0")
                    print("这可能意味着:")
                    print("1. 数据没有标注")
                    print("2. 需要特殊的解码方式")
                    print("3. 标签存储在其他字段")
                else:
                    print(f"\n✅ 成功提取到 {len(unique_labels)} 个不同的标签")
                    
                    # 根据Toronto 3D的类别映射显示标签含义
                    class_names = {
                        0: "Unclassified",
                        1: "Ground", 
                        2: "Road_markings",
                        3: "Natural",
                        4: "Building",
                        5: "Utility_line", 
                        6: "Pole",
                        7: "Car",
                        8: "Fence"
                    }
                    
                    print(f"\n标签含义:")
                    for label in unique_labels:
                        if label in class_names:
                            print(f"  {label}: {class_names[label]}")
                        else:
                            print(f"  {label}: Unknown")
            else:
                print(f"\n❌ 未找到scalar_Label字段")
                print(f"可用字段: {list(field_names)}")
                
        except Exception as e:
            print(f"访问数据失败: {e}")
            # 尝试其他方式
            print("尝试其他方式获取字段信息...")
            if hasattr(vertex_data, 'dtype'):
                print(f"vertex_data.dtype: {vertex_data.dtype}")
            if hasattr(vertex_data, 'properties'):
                print(f"vertex_data.properties: {vertex_data.properties}")
        
        # 旧的检查方式（作为备选）
        field_names = None
        if hasattr(vertex_data, 'dtype') and hasattr(vertex_data.dtype, 'names'):
            field_names = vertex_data.dtype.names
        
        if field_names and 'scalar_Label' in field_names:
            labels = vertex_data['scalar_Label']
            print(f"\nscalar_Label字段信息:")
            print(f"  数据类型: {labels.dtype}")
            print(f"  数组形状: {labels.shape}")
            print(f"  值范围: {labels.min()} ~ {labels.max()}")
            
            # 详细分析标签分布
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"  唯一标签: {unique_labels}")
            print(f"  标签数量: {len(unique_labels)}")
            
            # 显示标签分布
            print(f"\n标签分布:")
            for label, count in zip(unique_labels, counts):
                percentage = count / len(labels) * 100
                print(f"  标签 {label}: {count:,} 个点 ({percentage:.2f}%)")
            
            # 检查前100个点的标签值作为样本
            print(f"\n前10个点的标签值: {labels[:10]}")
            
            # 如果所有标签都是0，可能需要特殊处理
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                print("\n⚠️ 警告: 所有标签都是0")
                print("这可能意味着:")
                print("1. 数据没有标注")
                print("2. 需要特殊的解码方式")
                print("3. 标签存储在其他字段")
            else:
                print(f"\n✅ 成功提取到 {len(unique_labels)} 个不同的标签")
                
                # 根据Toronto 3D的类别映射显示标签含义
                class_names = {
                    0: "Unclassified",
                    1: "Ground", 
                    2: "Road_markings",
                    3: "Natural",
                    4: "Building",
                    5: "Utility_line", 
                    6: "Pole",
                    7: "Car",
                    8: "Fence"
                }
                
                print(f"\n标签含义:")
                for label in unique_labels:
                    if label in class_names:
                        print(f"  {label}: {class_names[label]}")
                    else:
                        print(f"  {label}: Unknown")
        else:
            print("\n❌ 未找到scalar_Label字段")
            
    except ImportError:
        print("需要安装plyfile库: pip install plyfile")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_label_extraction() 