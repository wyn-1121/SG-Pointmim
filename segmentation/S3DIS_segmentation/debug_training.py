"""
Toronto-3D训练问题诊断脚本
分析第一个epoch IoU为0的问题
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from data_utils.Toronto3DDataLoader import Toronto3DDataset
from models.toronto3d_semseg import get_model, get_loss

def check_data_distribution():
    """检查数据标签分布"""
    print("=" * 50)
    print("1. 数据标签分布检查")
    print("=" * 50)
    
    dataset = Toronto3DDataset(split='train', data_root='../data/Toronto_3D_processed', max_files=2)
    
    # 统计所有标签
    all_labels = []
    for i in range(min(10, len(dataset))):  # 检查前10个批次
        try:
            points, labels = dataset[i]
            all_labels.extend(labels.flatten())
            print(f"批次 {i}: 点数={points.shape}, 标签形状={labels.shape}")
            print(f"  标签范围: {labels.min()}-{labels.max()}")
            print(f"  标签分布: {np.bincount(labels.flatten(), minlength=9)}")
        except Exception as e:
            print(f"批次 {i} 错误: {e}")
    
    if all_labels:
        all_labels = np.array(all_labels)
        print(f"\n总体标签统计:")
        print(f"  标签范围: {all_labels.min()}-{all_labels.max()}")
        print(f"  标签计数: {np.bincount(all_labels, minlength=9)}")
        print(f"  标签权重: {dataset.labelweights}")

def check_model_output():
    """检查模型输出"""
    print("\n" + "=" * 50)
    print("2. 模型输出检查")
    print("=" * 50)
    
    # 创建模型
    model = get_model(9).cuda()
    criterion = get_loss().cuda()
    
    # 创建测试数据
    batch_size = 2
    num_points = 4096
    test_input = torch.randn(batch_size, 7, num_points).cuda()
    test_labels = torch.randint(0, 9, (batch_size, num_points)).cuda()
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"标签形状: {test_labels.shape}")
        
        # 检查输出值范围
        print(f"输出值范围: {output.min().item():.6f} - {output.max().item():.6f}")
        print(f"输出均值: {output.mean().item():.6f}")
        print(f"输出标准差: {output.std().item():.6f}")
        
        # 检查预测分布
        predictions = output.argmax(dim=-1)
        print(f"预测标签范围: {predictions.min().item()} - {predictions.max().item()}")
        pred_counts = torch.bincount(predictions.flatten(), minlength=9)
        print(f"预测分布: {pred_counts.cpu().numpy()}")
        
        # 计算损失
        output_flat = output.view(-1, 9)
        labels_flat = test_labels.view(-1)
        loss = criterion(output_flat, labels_flat)
        print(f"测试损失: {loss.item():.6f}")

def check_real_batch():
    """检查真实批次数据"""
    print("\n" + "=" * 50)
    print("3. 真实批次检查")
    print("=" * 50)
    
    dataset = Toronto3DDataset(split='train', data_root='../data/Toronto_3D_processed', max_files=2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    model = get_model(9).cuda()
    criterion = get_loss().cuda()
    weights = torch.Tensor(dataset.labelweights).cuda()
    
    for i, (points, target) in enumerate(dataloader):
        if i >= 1:  # 只检查第一个批次
            break
            
        print(f"批次 {i}:")
        print(f"  输入点云形状: {points.shape}")
        print(f"  标签形状: {target.shape}")
        print(f"  输入数值范围: {points.min().item():.6f} - {points.max().item():.6f}")
        print(f"  标签范围: {target.min().item()} - {target.max().item()}")
        
        # 检查每个维度的统计
        for dim in range(points.shape[1]):
            print(f"  维度{dim}: 均值={points[:, dim, :].mean().item():.6f}, "
                  f"标准差={points[:, dim, :].std().item():.6f}")
        
        # 检查标签分布
        target_flat = target.flatten()
        label_counts = torch.bincount(target_flat, minlength=9)
        print(f"  标签分布: {label_counts.numpy()}")
        
        # 转换数据格式并前向传播
        points = points.float().cuda()
        target = target.long().cuda()
        
        model.eval()
        with torch.no_grad():
            try:
                seg_pred = model(points)
                print(f"  模型输出形状: {seg_pred.shape}")
                print(f"  输出数值范围: {seg_pred.min().item():.6f} - {seg_pred.max().item():.6f}")
                
                # 计算损失
                seg_pred_flat = seg_pred.contiguous().view(-1, 9)
                target_flat = target.view(-1)
                
                # 检查损失计算
                loss_no_weight = criterion(seg_pred_flat, target_flat, None, None)
                loss_with_weight = criterion(seg_pred_flat, target_flat, None, weights)
                print(f"  无权重损失: {loss_no_weight.item():.6f}")
                print(f"  带权重损失: {loss_with_weight.item():.6f}")
                
                # 检查预测结果
                predictions = seg_pred.argmax(dim=-1)
                pred_flat = predictions.view(-1)
                
                # 计算准确率
                correct = (pred_flat == target_flat).sum().item()
                total = target_flat.size(0)
                accuracy = correct / total
                print(f"  准确率: {accuracy:.6f} ({correct}/{total})")
                
                # 检查预测分布
                pred_counts = torch.bincount(pred_flat, minlength=9)
                print(f"  预测分布: {pred_counts.cpu().numpy()}")
                
                # 计算IoU
                iou_per_class = []
                for cls in range(9):
                    tp = ((pred_flat == cls) & (target_flat == cls)).sum().item()
                    fp = ((pred_flat == cls) & (target_flat != cls)).sum().item() 
                    fn = ((pred_flat != cls) & (target_flat == cls)).sum().item()
                    
                    if tp + fp + fn > 0:
                        iou = tp / (tp + fp + fn)
                    else:
                        iou = 0
                    iou_per_class.append(iou)
                    print(f"  类别{cls} IoU: {iou:.6f} (TP={tp}, FP={fp}, FN={fn})")
                
                mean_iou = np.mean(iou_per_class)
                print(f"  平均IoU: {mean_iou:.6f}")
                
            except Exception as e:
                print(f"  模型前向传播错误: {e}")
                import traceback
                traceback.print_exc()

def check_loss_function():
    """检查损失函数实现"""
    print("\n" + "=" * 50)
    print("4. 损失函数检查")
    print("=" * 50)
    
    # 创建测试数据
    batch_size = 2
    num_points = 100
    num_classes = 9
    
    # 创建log_softmax输出（模型的实际输出）
    logits = torch.randn(batch_size * num_points, num_classes)
    log_probs = F.log_softmax(logits, dim=1)
    
    # 创建标签
    labels = torch.randint(0, num_classes, (batch_size * num_points,))
    
    # 创建权重
    weights = torch.ones(num_classes)
    weights[0] = 5.0  # 给第一个类别更高权重
    
    print(f"输入形状: {log_probs.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"权重: {weights}")
    
    # 测试损失函数
    criterion = get_loss()
    
    loss_no_weight = criterion(log_probs, labels, None, None)
    loss_with_weight = criterion(log_probs, labels, None, weights)
    
    print(f"无权重损失: {loss_no_weight.item():.6f}")
    print(f"带权重损失: {loss_with_weight.item():.6f}")
    
    # 比较PyTorch原生实现
    pytorch_loss_no_weight = F.nll_loss(log_probs, labels)
    pytorch_loss_with_weight = F.nll_loss(log_probs, labels, weight=weights)
    
    print(f"PyTorch无权重损失: {pytorch_loss_no_weight.item():.6f}")
    print(f"PyTorch带权重损失: {pytorch_loss_with_weight.item():.6f}")

def main():
    """主函数"""
    print("Toronto-3D训练问题诊断")
    print("分析第一个epoch IoU为0的问题")
    
    try:
        check_data_distribution()
    except Exception as e:
        print(f"数据分布检查失败: {e}")
    
    try:
        check_model_output()
    except Exception as e:
        print(f"模型输出检查失败: {e}")
    
    try:
        check_real_batch()
    except Exception as e:
        print(f"真实批次检查失败: {e}")
    
    try:
        check_loss_function()
    except Exception as e:
        print(f"损失函数检查失败: {e}")

if __name__ == "__main__":
    main() 