import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import sys
import importlib
import random
from tqdm import tqdm


# =========================================================
#                      核心配置
# =========================================================

TARGET_CATEGORIES = ['Airplane', 'Chair', 'Motorbike', 'Lamp']

DATA_ROOT = '/home/CX/wyn/sg_pointmim/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

CKPT_BASELINE = 'log/pointmae_baseline/checkpoints/part_seg.pth'
CKPT_OURS = 'log/part_seg/exp/checkpoints/best_model.pth'

MODEL_NAME = 'pt'
OUTPUT_NAME = 'Fig3_PartSeg_Resampled.png'

POINT_SIZE = 18
SHOW_ERROR_MAP = False
RAINBOW_AXIS = 'x'

SEED = 42

# 搜索范围越大，越容易找到视觉差异明显的样本
MAX_SEARCH_NUM = 3000

# 最低要求。只是过滤极端坏样本，不用于图上显示
MIN_OURS_ACC = 0.80
MIN_OURS_MIOU = 0.65

# 候选样本至少要求 ours 有一定优势
MIN_ACC_GAP = 0.04
MIN_IOU_GAP = 0.04


# =========================================================
#                 ShapeNetPart 类别和 part 映射
# =========================================================
# ShapeNetPart 50 个 part label 的常用类别映射
# 如果你的 dataset 内部 label 顺序不同，建议以 dataset 的 seg_classes 为准。

SEG_CLASSES = {
    'Airplane': [0, 1, 2, 3],
    'Bag': [4, 5],
    'Cap': [6, 7],
    'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15],
    'Earphone': [16, 17, 18],
    'Guitar': [19, 20, 21],
    'Knife': [22, 23],
    'Lamp': [24, 25, 26, 27],
    'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Mug': [36, 37],
    'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43],
    'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49],
}


# =========================================================
#                    不同类别的观察角度
# =========================================================

VIEW_CONFIG = {
    'Airplane':  {'elev': 90, 'azim': -90},
    'Chair':     {'elev': 20, 'azim': 30},
    'Motorbike': {'elev': 10, 'azim': 90},
    'Lamp':      {'elev': 20, 'azim': 120},
    'Table':     {'elev': 20, 'azim': 30},
    'default':   {'elev': 20, 'azim': 30}
}


# =========================================================
#                    环境和随机种子
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(ROOT_DIR)

try:
    from dataset import PartNormalDataset
    MODEL = importlib.import_module(MODEL_NAME)
except ImportError as e:
    try:
        sys.path.append(os.path.join(ROOT_DIR, 'segmentation'))
        from data.ShapeNetPart import PartNormalDataset
        MODEL = importlib.import_module(MODEL_NAME)
    except Exception:
        print(f"环境错误: {e}")
        print("请确认你在 segmentation/ 目录下运行此脚本。")
        sys.exit(1)


# =========================================================
#                  论文风格部件分割配色
# =========================================================

PART_COLORS = np.array([
    [0.894, 0.102, 0.110],
    [0.216, 0.494, 0.722],
    [0.302, 0.686, 0.290],
    [0.596, 0.306, 0.639],
    [1.000, 0.498, 0.000],
    [1.000, 1.000, 0.200],
    [0.651, 0.337, 0.157],
    [0.969, 0.506, 0.749],
    [0.600, 0.600, 0.600],
    [0.121, 0.470, 0.705],
    [0.682, 0.780, 0.910],
    [0.737, 0.741, 0.133],
    [0.090, 0.745, 0.811],
    [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761],
    [0.498, 0.498, 0.498],
    [0.400, 0.760, 0.647],
    [0.988, 0.553, 0.384],
    [0.553, 0.627, 0.796],
    [0.905, 0.541, 0.765],
    [0.651, 0.847, 0.329],
    [1.000, 0.851, 0.184],
    [0.898, 0.768, 0.580],
    [0.702, 0.702, 0.702],
], dtype=np.float32)


def get_colors(labels, mode='part', gt=None):
    labels = labels.astype(int)

    if mode == 'error' and gt is not None:
        colors = np.ones((len(labels), 3), dtype=np.float32) * 0.82
        errors = labels != gt
        colors[errors] = [0.95, 0.10, 0.10]
        return colors

    return PART_COLORS[labels % len(PART_COLORS)]


def get_rainbow(points):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis = axis_map.get(RAINBOW_AXIS, 0)

    vals = points[:, axis]
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    return plt.cm.turbo(vals)[:, :3]


# =========================================================
#                    指标计算
# =========================================================

def part_iou(pred, gt, part_ids):
    """
    单个样本的 class-level part mIoU。
    只在该类别允许的 part_ids 内计算。
    """
    ious = []

    for part_id in part_ids:
        pred_mask = pred == part_id
        gt_mask = gt == part_id

        union = np.logical_or(pred_mask, gt_mask).sum()
        inter = np.logical_and(pred_mask, gt_mask).sum()

        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter / float(union))

    return float(np.mean(ious))


def point_acc(pred, gt):
    return float(np.mean(pred == gt))


def boundary_error_ratio(pred, gt, points, k=12):
    """
    一个辅助筛选指标：希望 baseline 的错误不是随机零散，而是在复杂区域有差异。
    这个指标不是必须精确，只用于挑图。
    返回错误点比例。
    """
    return float(np.mean(pred != gt))


# =========================================================
#                    模型加载函数
# =========================================================

def resolve_path(path):
    if os.path.exists(path):
        return path

    p1 = os.path.join(BASE_DIR, path)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(ROOT_DIR, path)
    if os.path.exists(p2):
        return p2

    return path


def load_checkpoint(model, path):
    path = resolve_path(path)

    if not os.path.exists(path):
        print(f"权重未找到: {path}")
        return False

    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')

    if isinstance(ckpt, dict):
        state = ckpt.get(
            'model_state_dict',
            ckpt.get('model', ckpt.get('base_model', ckpt))
        )
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '')
        new_state[nk] = v

    msg = model.load_state_dict(new_state, strict=False)

    print("Missing keys:", len(msg.missing_keys))
    print("Unexpected keys:", len(msg.unexpected_keys))

    model.cuda()
    model.eval()

    return True


def load_model(ckpt_path):
    model = MODEL.get_model(50).cuda()
    success = load_checkpoint(model, ckpt_path)
    if not success:
        return None
    return model


# =========================================================
#                    推理函数
# =========================================================

def forward_model(model, pts, lbl_hot):
    logits = model(pts, lbl_hot)

    if logits.dim() == 3 and logits.shape[1] == 50:
        logits = logits.transpose(2, 1)

    pred = logits.argmax(-1)
    return pred


# =========================================================
#                    样本重新筛选
# =========================================================

def find_best_sample(category, model_base, model_ours):
    print(f"\n>>> 重新筛选 [{category}] 样本...")

    try:
        ds = PartNormalDataset(
            root=DATA_ROOT,
            npoints=2048,
            split='test',
            class_choice=category,
            normal_channel=False
        )
    except Exception as e:
        print(f"无法加载类别 {category}: {e}")
        return None

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    part_ids = SEG_CLASSES[category]

    best = None
    best_score = -999.0

    candidate_count = 0

    for i, batch in enumerate(tqdm(loader, desc=f"Searching {category}")):
        if i >= MAX_SEARCH_NUM:
            break

        if len(batch) != 3:
            print("数据集返回格式不是 (points, label, target)，请检查 PartNormalDataset。")
            return None

        pts, lbl, tgt = batch

        pts = pts.float().cuda()       # [B, N, 3]
        tgt = tgt.long().cuda()        # [B, N]
        lbl = lbl.long()               # [B]

        pts_input = pts.transpose(2, 1)
        lbl_hot = torch.eye(16)[lbl.cpu().numpy()].float().cuda()

        with torch.no_grad():
            pred_ours = forward_model(model_ours, pts_input, lbl_hot)
            pred_base = forward_model(model_base, pts_input, lbl_hot)

        pred_ours_np = pred_ours.cpu().numpy()[0]
        pred_base_np = pred_base.cpu().numpy()[0]
        gt_np = tgt.cpu().numpy()[0]
        pts_np = pts.cpu().numpy()[0]

        acc_o = point_acc(pred_ours_np, gt_np)
        acc_b = point_acc(pred_base_np, gt_np)

        miou_o = part_iou(pred_ours_np, gt_np, part_ids)
        miou_b = part_iou(pred_base_np, gt_np, part_ids)

        acc_gap = acc_o - acc_b
        iou_gap = miou_o - miou_b

        base_err = boundary_error_ratio(pred_base_np, gt_np, pts_np)
        ours_err = boundary_error_ratio(pred_ours_np, gt_np, pts_np)

        # 过滤：Ours 不能太差，同时需要有一定提升
        if acc_o < MIN_OURS_ACC:
            continue
        if miou_o < MIN_OURS_MIOU:
            continue
        if acc_gap < MIN_ACC_GAP and iou_gap < MIN_IOU_GAP:
            continue

        candidate_count += 1

        # 综合得分：
        # 1. 优先 IoU 提升；
        # 2. 兼顾 acc 提升；
        # 3. 希望 baseline 错误多，ours 错误少；
        # 4. 轻微偏好 ours 本身质量高。
        score = (
            0.60 * iou_gap +
            0.35 * acc_gap +
            0.15 * (base_err - ours_err) +
            0.05 * miou_o
        )

        if score > best_score:
            best_score = score
            best = {
                'points': pts_np,
                'base': pred_base_np,
                'ours': pred_ours_np,
                'gt': gt_np,
                'acc_b': acc_b,
                'acc_o': acc_o,
                'miou_b': miou_b,
                'miou_o': miou_o,
                'acc_gap': acc_gap,
                'iou_gap': iou_gap,
                'score': score,
                'idx': i
            }

    if best is None:
        print(f"  Warning: {category} 未找到满足阈值的样本，降低阈值后再试。")
        return None

    print(
        f"  [选中] {category} | ID:{best['idx']} | "
        f"Score:{best['score']:.4f} | "
        f"Acc Base/Ours:{best['acc_b']:.2%}/{best['acc_o']:.2%} | "
        f"mIoU Base/Ours:{best['miou_b']:.2%}/{best['miou_o']:.2%} | "
        f"Gap Acc/IoU:{best['acc_gap']:+.2%}/{best['iou_gap']:+.2%} | "
        f"Candidates:{candidate_count}"
    )

    return best


# =========================================================
#                    绘图函数
# =========================================================

def normalize_points(pts):
    pts = pts.copy()
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / (scale + 1e-8)
    return pts


def plot_one_cloud(ax, pts, colors, view, point_size=18):
    ax.set_axis_off()
    ax.view_init(view['elev'], view['azim'])

    lim = 0.85
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors,
        s=point_size,
        edgecolors='none',
        alpha=0.98,
        depthshade=False
    )


def plot_grid(samples, save_name):
    n = len(samples)

    if n == 0:
        print("没有可绘制样本。")
        return

    fig = plt.figure(figsize=(16, 3.6 * n), dpi=300)

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.94,
        bottom=0.03,
        wspace=-0.12,
        hspace=-0.05
    )

    column_titles = [
        '(a) Input',
        '(b) Point-MAE',
        '(c) SG-PointMIM',
        '(d) Ground Truth'
    ]

    for i, (cat, d) in enumerate(samples.items()):
        pts = normalize_points(d['points'])
        view = VIEW_CONFIG.get(cat, VIEW_CONFIG['default'])

        configs = [
            ('Input', get_rainbow(pts), column_titles[0]),
            ('Point-MAE', get_colors(d['base'], 'error' if SHOW_ERROR_MAP else 'part', d['gt']), column_titles[1]),
            ('SG-PointMIM', get_colors(d['ours'], 'error' if SHOW_ERROR_MAP else 'part', d['gt']), column_titles[2]),
            ('Ground Truth', get_colors(d['gt'], 'part'), column_titles[3])
        ]

        for j, (_, colors, title) in enumerate(configs):
            ax = fig.add_subplot(n, 4, i * 4 + j + 1, projection='3d')

            plot_one_cloud(
                ax=ax,
                pts=pts,
                colors=colors,
                view=view,
                point_size=POINT_SIZE
            )

            if i == 0:
                ax.set_title(
                    title,
                    fontsize=18,
                    fontweight='bold',
                    y=1.02
                )

            if j == 0:
                ax.text2D(
                    -0.08,
                    0.5,
                    cat,
                    transform=ax.transAxes,
                    fontsize=18,
                    fontweight='bold',
                    rotation=90,
                    va='center',
                    ha='center'
                )

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.03)
    plt.close()

    print(f"\n>>> 图片已生成: {save_name}")


# =========================================================
#                         主函数
# =========================================================

def main():
    print(">>> Loading Models...")

    model_ours = load_model(CKPT_OURS)
    if model_ours is None:
        raise FileNotFoundError(f"Ours 权重无法加载: {CKPT_OURS}")

    model_base = load_model(CKPT_BASELINE)
    if model_base is None:
        raise FileNotFoundError(f"Baseline 权重无法加载: {CKPT_BASELINE}")

    print("\n>>> Start resampling visualization examples...")

    data = {}

    for cat in TARGET_CATEGORIES:
        res = find_best_sample(cat, model_base, model_ours)
        if res is not None:
            data[cat] = res
        else:
            print(f"Warning: {cat} 没有找到合适样本。")

    if len(data) > 0:
        print("\n>>> Final selected samples:")
        for cat, d in data.items():
            print(
                f"{cat}: ID={d['idx']}, "
                f"Base Acc={d['acc_b']:.2%}, Ours Acc={d['acc_o']:.2%}, "
                f"Base mIoU={d['miou_b']:.2%}, Ours mIoU={d['miou_o']:.2%}, "
                f"Acc Gap={d['acc_gap']:+.2%}, IoU Gap={d['iou_gap']:+.2%}"
            )

        plot_grid(data, OUTPUT_NAME)
    else:
        print("未找到任何符合条件的样本，无法绘制。")
        print("建议降低 MIN_ACC_GAP / MIN_IOU_GAP 或增大 MAX_SEARCH_NUM。")


if __name__ == '__main__':
    main()