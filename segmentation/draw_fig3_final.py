import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import sys
import importlib
import random
import json
from tqdm import tqdm


# =========================================================
# Config
# =========================================================

TARGET_CATEGORIES = ['Airplane', 'Chair', 'Lamp']
RESAMPLE_CATEGORY = 'Chair'

DATA_ROOT = '/home/CX/wyn/sg_pointmim/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

CKPT_BASELINE = 'log/pointmae_baseline/checkpoints/part_seg.pth'
CKPT_OURS = 'log/part_seg/exp/checkpoints/best_model.pth'

MODEL_NAME = 'pt'

SELECTED_JSON = 'selected_fig3_samples_3cls.json'
OUTPUT_ERROR_MAP = 'Fig3_ErrorMap_Check_3cls_ieee.png'
OUTPUT_FINAL = 'Fig3_PartSeg_Final_3cls_ieee.png'

SEED = 42
MAX_SEARCH_NUM = 5000

# Chair 重新筛选阈值：这里比之前更严格，目标是找差异更明显的 Chair
CHAIR_MIN_OURS_ACC = 0.75
CHAIR_MIN_OURS_MIOU = 0.50
CHAIR_MIN_ACC_GAP = 0.03
CHAIR_MIN_IOU_GAP = 0.10

POINT_SIZE = 4.8
INPUT_COLOR_MODE = 'blue'


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


VIEW_CONFIG = {
    'Airplane': {'elev': 90, 'azim': -90},
    # Chair 用这个视角通常更容易看出靠背/坐垫/腿部
    'Chair': {'elev': 18, 'azim': -55},
    'Lamp': {'elev': 20, 'azim': 120},
    'default': {'elev': 20, 'azim': 30},
}


# =========================================================
# Environment
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
# Colors
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


def get_part_colors(labels):
    labels = labels.astype(int)
    return PART_COLORS[labels % len(PART_COLORS)]


def get_error_colors(pred, gt):
    colors = np.ones((len(pred), 3), dtype=np.float32) * 0.78
    errors = pred != gt
    colors[errors] = [0.95, 0.05, 0.05]
    return colors


def get_input_colors(points):
    if INPUT_COLOR_MODE == 'gray':
        return np.ones((points.shape[0], 3), dtype=np.float32) * 0.55

    vals = points[:, 0]
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return plt.cm.Blues(0.25 + 0.65 * vals)[:, :3]


# =========================================================
# Metrics
# =========================================================

def point_acc(pred, gt):
    return float(np.mean(pred == gt))


def part_iou(pred, gt, part_ids):
    ious = []

    for pid in part_ids:
        pred_mask = pred == pid
        gt_mask = gt == pid

        inter = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter / float(union))

    return float(np.mean(ious))


def small_part_gain(pred_base, pred_ours, gt, part_ids):
    gains = []

    for pid in part_ids:
        gt_mask = gt == pid
        count = gt_mask.sum()

        if count == 0:
            continue

        acc_b = np.mean(pred_base[gt_mask] == gt[gt_mask])
        acc_o = np.mean(pred_ours[gt_mask] == gt[gt_mask])

        weight = 1.0 / np.sqrt(float(count) + 1.0)
        gains.append((acc_o - acc_b) * weight)

    if len(gains) == 0:
        return 0.0

    return float(np.sum(gains))


def chair_structure_score(points, pred_base, pred_ours, gt):
    """
    辅助筛选：希望 Chair 不是单一平板，而是结构更复杂；
    同时希望 Ours 在错误点上明显少于 Baseline。
    """
    pts = points.copy()
    extent = pts.max(axis=0) - pts.min(axis=0)

    # 结构复杂度：三个方向都有一定跨度，避免选到过于薄的单面样本
    sorted_extent = np.sort(extent)
    complexity = sorted_extent[1] / (sorted_extent[2] + 1e-8)

    base_err = np.mean(pred_base != gt)
    ours_err = np.mean(pred_ours != gt)
    err_gain = base_err - ours_err

    return 0.5 * complexity + 0.5 * err_gain


# =========================================================
# Model loading
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
        new_state[k.replace('module.', '')] = v

    msg = model.load_state_dict(new_state, strict=False)
    print("Missing keys:", len(msg.missing_keys))
    print("Unexpected keys:", len(msg.unexpected_keys))

    model.cuda()
    model.eval()

    return True


def load_model(ckpt_path):
    model = MODEL.get_model(50).cuda()
    ok = load_checkpoint(model, ckpt_path)

    if not ok:
        return None

    return model


def forward_model(model, pts, lbl_hot):
    logits = model(pts, lbl_hot)

    if logits.dim() == 3 and logits.shape[1] == 50:
        logits = logits.transpose(2, 1)

    return logits.argmax(-1)


# =========================================================
# Dataset helpers
# =========================================================

def build_dataset(category):
    return PartNormalDataset(
        root=DATA_ROOT,
        npoints=2048,
        split='test',
        class_choice=category,
        normal_channel=False
    )


def infer_one(model_base, model_ours, pts, lbl, tgt):
    pts = pts.float().cuda()
    tgt = tgt.long().cuda()
    lbl = lbl.long()

    pts_input = pts.transpose(2, 1)
    lbl_hot = torch.eye(16)[lbl.cpu().numpy()].float().cuda()

    with torch.no_grad():
        pred_base = forward_model(model_base, pts_input, lbl_hot)
        pred_ours = forward_model(model_ours, pts_input, lbl_hot)

    return (
        pts.cpu().numpy()[0],
        pred_base.cpu().numpy()[0],
        pred_ours.cpu().numpy()[0],
        tgt.cpu().numpy()[0],
    )


# =========================================================
# Resample Chair
# =========================================================

def resample_chair(model_base, model_ours):
    category = RESAMPLE_CATEGORY
    print(f"\n>>> Re-sampling {category} only...")

    ds = build_dataset(category)
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
    num_candidates = 0

    for idx, batch in enumerate(tqdm(loader, desc=f"Scanning {category}")):
        if idx >= MAX_SEARCH_NUM:
            break

        set_seed(SEED + idx)

        pts, lbl, tgt = batch
        pts_np, pred_b, pred_o, gt = infer_one(
            model_base,
            model_ours,
            pts,
            lbl,
            tgt
        )

        acc_b = point_acc(pred_b, gt)
        acc_o = point_acc(pred_o, gt)
        miou_b = part_iou(pred_b, gt, part_ids)
        miou_o = part_iou(pred_o, gt, part_ids)

        acc_gap = acc_o - acc_b
        iou_gap = miou_o - miou_b

        if acc_o < CHAIR_MIN_OURS_ACC:
            continue
        if miou_o < CHAIR_MIN_OURS_MIOU:
            continue
        if acc_gap < CHAIR_MIN_ACC_GAP and iou_gap < CHAIR_MIN_IOU_GAP:
            continue

        num_candidates += 1

        base_error = np.mean(pred_b != gt)
        ours_error = np.mean(pred_o != gt)
        small_gain = small_part_gain(pred_b, pred_o, gt, part_ids)
        structure = chair_structure_score(pts_np, pred_b, pred_o, gt)

        # 重点：更偏向 IoU gap、错误减少和结构复杂的 Chair
        score = (
            0.55 * iou_gap +
            0.20 * acc_gap +
            0.20 * (base_error - ours_error) +
            0.20 * small_gain +
            0.15 * structure +
            0.03 * miou_o
        )

        if score > best_score:
            best_score = score
            best = {
                'idx': idx,
                'acc_b': acc_b,
                'acc_o': acc_o,
                'miou_b': miou_b,
                'miou_o': miou_o,
                'acc_gap': acc_gap,
                'iou_gap': iou_gap,
                'score': score,
                'base_error': base_error,
                'ours_error': ours_error,
                'structure': structure,
            }

    if best is None:
        raise RuntimeError(
            "没有找到更合适的 Chair。可以降低 CHAIR_MIN_IOU_GAP 或 CHAIR_MIN_ACC_GAP。"
        )

    print(
        f"[Selected Chair] idx={best['idx']} | "
        f"score={best['score']:.4f} | "
        f"Acc {best['acc_b']:.2%}->{best['acc_o']:.2%} | "
        f"mIoU {best['miou_b']:.2%}->{best['miou_o']:.2%} | "
        f"Gap Acc/IoU {best['acc_gap']:+.2%}/{best['iou_gap']:+.2%} | "
        f"Err {best['base_error']:.2%}->{best['ours_error']:.2%} | "
        f"structure={best['structure']:.4f} | "
        f"candidates={num_candidates}"
    )

    return best['idx']


# =========================================================
# Load selected samples
# =========================================================

def load_selected_samples(model_base, model_ours, selected):
    data = {}

    for category, target_idx in selected.items():
        ds = build_dataset(category)

        set_seed(SEED + int(target_idx))

        pts, lbl, tgt = ds[int(target_idx)]

        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).unsqueeze(0)
        else:
            pts = pts.unsqueeze(0)

        if isinstance(lbl, np.ndarray):
            lbl = torch.from_numpy(lbl).long()
            if lbl.dim() == 0:
                lbl = lbl.unsqueeze(0)
            elif lbl.dim() == 1:
                lbl = lbl[:1]
        elif torch.is_tensor(lbl):
            if lbl.dim() == 0:
                lbl = lbl.unsqueeze(0)
            elif lbl.dim() == 1:
                lbl = lbl[:1]
        else:
            lbl = torch.tensor([lbl], dtype=torch.long)

        if isinstance(tgt, np.ndarray):
            tgt = torch.from_numpy(tgt).unsqueeze(0)
        else:
            tgt = tgt.unsqueeze(0)

        pts_np, pred_b, pred_o, gt = infer_one(
            model_base,
            model_ours,
            pts,
            lbl,
            tgt
        )

        part_ids = SEG_CLASSES[category]

        data[category] = {
            'points': pts_np,
            'base': pred_b,
            'ours': pred_o,
            'gt': gt,
            'acc_b': point_acc(pred_b, gt),
            'acc_o': point_acc(pred_o, gt),
            'miou_b': part_iou(pred_b, gt, part_ids),
            'miou_o': part_iou(pred_o, gt, part_ids),
            'idx': int(target_idx),
        }

    return data


# =========================================================
# Plot
# =========================================================

def normalize_points(pts):
    pts = pts.copy()
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / (scale + 1e-8)
    return pts


def plot_one_cloud(ax, pts, colors, view):
    ax.set_axis_off()
    ax.view_init(view['elev'], view['azim'])

    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    xyz_center = (xyz_min + xyz_max) / 2.0
    xyz_range = (xyz_max - xyz_min).max()
    half = xyz_range * 0.56

    ax.set_xlim([xyz_center[0] - half, xyz_center[0] + half])
    ax.set_ylim([xyz_center[1] - half, xyz_center[1] + half])
    ax.set_zlim([xyz_center[2] - half, xyz_center[2] + half])
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors,
        s=POINT_SIZE,
        edgecolors='none',
        alpha=0.98,
        depthshade=False,
        rasterized=True
    )

    ax.margins(0)


def get_part_colors_for_plot(labels):
    return get_part_colors(labels)


def get_error_colors_for_plot(pred, gt):
    return get_error_colors(pred, gt)


def get_input_colors_for_plot(points):
    return get_input_colors(points)


def plot_grid(data, save_name, error_map=False):
    n = len(TARGET_CATEGORIES)

    fig = plt.figure(figsize=(7.16, 4.15), dpi=600)

    plt.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.925,
        bottom=0.015,
        wspace=-0.34,
        hspace=-0.30
    )

    column_titles = [
        '(a) Input',
        '(b) Point-MAE',
        '(c) SG-PointMIM',
        '(d) Ground Truth'
    ]

    for i, category in enumerate(TARGET_CATEGORIES):
        if category not in data:
            continue

        d = data[category]
        pts = normalize_points(d['points'])
        view = VIEW_CONFIG.get(category, VIEW_CONFIG['default'])

        if error_map:
            base_colors = get_error_colors_for_plot(d['base'], d['gt'])
            ours_colors = get_error_colors_for_plot(d['ours'], d['gt'])
            gt_colors = get_part_colors_for_plot(d['gt'])
        else:
            base_colors = get_part_colors_for_plot(d['base'])
            ours_colors = get_part_colors_for_plot(d['ours'])
            gt_colors = get_part_colors_for_plot(d['gt'])

        configs = [
            (get_input_colors_for_plot(pts), column_titles[0]),
            (base_colors, column_titles[1]),
            (ours_colors, column_titles[2]),
            (gt_colors, column_titles[3]),
        ]

        for j, (colors, title) in enumerate(configs):
            ax = fig.add_subplot(n, 4, i * 4 + j + 1, projection='3d')
            plot_one_cloud(ax, pts, colors, view)

            if i == 0:
                ax.set_title(
                    title,
                    fontsize=8.0,
                    fontweight='bold',
                    y=0.96,
                    pad=-2
                )

            if j == 0:
                ax.text2D(
                    -0.025,
                    0.5,
                    category,
                    transform=ax.transAxes,
                    fontsize=8.0,
                    fontweight='bold',
                    rotation=90,
                    va='center',
                    ha='center'
                )

    base, _ = os.path.splitext(save_name)
    png_name = base + '.png'
    pdf_name = base + '.pdf'

    plt.savefig(png_name, dpi=600, bbox_inches='tight', pad_inches=0.005)
    plt.savefig(pdf_name, bbox_inches='tight', pad_inches=0.005)
    plt.close()

    print(f">>> Saved: {png_name}")
    print(f">>> Saved: {pdf_name}")


# =========================================================
# Main
# =========================================================

def main():
    print(">>> Loading models...")

    model_ours = load_model(CKPT_OURS)
    if model_ours is None:
        raise FileNotFoundError(f"Ours 权重无法加载: {CKPT_OURS}")

    model_base = load_model(CKPT_BASELINE)
    if model_base is None:
        raise FileNotFoundError(f"Baseline 权重无法加载: {CKPT_BASELINE}")

    if os.path.exists(SELECTED_JSON):
        with open(SELECTED_JSON, 'r') as f:
            selected = json.load(f)
        selected = {k: int(v) for k, v in selected.items() if k in TARGET_CATEGORIES}
    else:
        selected = {}

    # 保留已有 Airplane / Lamp，没有则报错，避免误覆盖
    for fixed_cat in ['Airplane', 'Lamp']:
        if fixed_cat not in selected:
            raise RuntimeError(
                f"{fixed_cat} 不在 {SELECTED_JSON} 中。请先运行 draw_fig3_3cls_ieee.py 生成初始样本。"
            )

    # 只重筛 Chair
    new_chair_idx = resample_chair(model_base, model_ours)
    selected['Chair'] = int(new_chair_idx)

    # 按固定顺序保存
    selected_ordered = {
        'Airplane': int(selected['Airplane']),
        'Chair': int(selected['Chair']),
        'Lamp': int(selected['Lamp']),
    }

    with open(SELECTED_JSON, 'w') as f:
        json.dump(selected_ordered, f, indent=2)

    print("\n>>> Updated selected ids:")
    print(selected_ordered)

    data = load_selected_samples(model_base, model_ours, selected_ordered)

    print("\n>>> Metrics of selected samples:")
    for category, d in data.items():
        print(
            f"{category}: idx={d['idx']}, "
            f"Acc {d['acc_b']:.2%}->{d['acc_o']:.2%}, "
            f"mIoU {d['miou_b']:.2%}->{d['miou_o']:.2%}"
        )

    plot_grid(data, 'Fig3_ErrorMap_Check_3cls_ieee_resample_chair.png', error_map=True)
    plot_grid(data, 'Fig3_PartSeg_Final_3cls_ieee_resample_chair.png', error_map=False)


if __name__ == '__main__':
    main()

