"""
Batch inference for all prepared samples under segmentation/data.

For each sample directory, finds a point cloud file (*.pts/*.txt/*.xyz/*.npy),
determines category from folder name (CN keywords mapped to ShapeNetPart names),
then runs the part segmentation model twice with two checkpoints and writes:
  - mae.seg   (from --ckpt_mae)
  - pred.seg  (from --ckpt_pred)

Usage:
  python segmentation/batch_infer_all.py \
    --data_root segmentation/data/ShapeNetPart \
    --ckpt_mae \
      "/mnt/sdb2/MDY/idea2/segmentation/log/mae.pth" \
    --ckpt_pred \
      "/mnt/sdb2/MDY/idea2/segmentation/log/part_seg/partseg1018/checkpoints/best_model.pth" \
    --gpu 0

Notes:
  - Requires CUDA extensions used by `segmentation/models/pt.py` to be available.
  - Category order matches the canonical 16-class order used in training.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
import importlib

# Local imports and paths
SEG_DIR = Path(__file__).parent
sys.path.append(str(SEG_DIR / 'models'))  # for model imports
from pointnet_util import pc_normalize  # type: ignore


# Canonical 16-class order (aligns with training)
CATS = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
    'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'
]

SEG_CLASSES = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
    'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
    'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
}

CAT2ID = {n: i for i, n in enumerate(CATS)}
ID2CAT = {i: n for n, i in CAT2ID.items()}

# Simple CN keyword mapping to canonical category
CN_KW_TO_CAT = {
    '飞机': 'Airplane',
    '耳机': 'Earphone',
    '摩托车': 'Motorbike',
    '手枪': 'Pistol',
    '台灯': 'Lamp',  # 灯/台灯
    '灯': 'Lamp',
    '桌子': 'Table',
    '椅子': 'Chair',
    '包': 'Bag',
    '帽': 'Cap',
    '汽车': 'Car',
    '吉他': 'Guitar',
    '刀': 'Knife',
    '电脑': 'Laptop',
    '杯': 'Mug',
    '火箭': 'Rocket',
    '滑板': 'Skateboard',
}


def to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    eye = torch.eye(num_classes, device=y.device)
    return eye[y.view(-1)]


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.npy':
        arr = np.load(path)
        pts = arr[:, :3].astype(np.float32)
    else:
        try:
            data = np.loadtxt(path, delimiter=',').astype(np.float32)
        except ValueError:
            data = np.loadtxt(path).astype(np.float32)
        if data.ndim == 1:
            data = data[None, :]
        pts = data[:, :3].astype(np.float32)
    return pts


def restrict_and_argmax(logits: torch.Tensor, cat_id: int) -> np.ndarray:
    # logits [1, N, 50] -> argmax within valid labels for the given category
    cat_name = ID2CAT[int(cat_id)]
    valid = SEG_CLASSES[cat_name]
    selected = logits[..., valid]
    local_idx = selected.argmax(dim=-1)  # [1, N]
    pred = (local_idx + valid[0]).squeeze(0).detach().cpu().numpy()
    return pred


def detect_category_from_folder(folder: Path) -> str | None:
    name = folder.name
    for kw, cat in CN_KW_TO_CAT.items():
        if kw in name:
            return cat
    return None


def build_and_load_model(ckpt_path: Path, device: torch.device):
    MODEL = importlib.import_module('pt')  # segmentation/models/pt.py
    model = MODEL.get_model(cls_dim=50).to(device)
    model.eval()

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        model.load_state_dict(state, strict=False)
    elif isinstance(ckpt, dict) and 'base_model' in ckpt:
        base_ckpt = {k.replace('module.', ''): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder'):
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]
        model.load_state_dict(base_ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model


def infer_and_save(model, pts_path: Path, cat_id: int, out_path: Path, device: torch.device):
    pts = load_points(pts_path)
    pts_norm = pc_normalize(pts.copy())
    tensor = torch.from_numpy(pts_norm).unsqueeze(0).float().to(device)  # [1, N, 3]
    tensor = tensor.transpose(2, 1)  # [1, 3, N]
    label = torch.tensor([cat_id], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tensor, to_categorical(label, 16))  # [1, N, 50]
        pred = restrict_and_argmax(logits, cat_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, pred.astype(np.int32), fmt='%d')


def main():
    parser = argparse.ArgumentParser(description='Batch infer ShapeNetPart samples')
    parser.add_argument('--data_root', type=str,
                        default=str(SEG_DIR / 'data' / 'ShapeNetPart'),
                        help='root containing sample folders')
    parser.add_argument('--ckpt_mae', type=str,
                        default='/mnt/sdb2/MDY/idea2/segmentation/log/mae.pth',
                        help='checkpoint for mae.seg')
    parser.add_argument('--ckpt_pred', type=str,
                        default='/mnt/sdb2/MDY/idea2/segmentation/log/part_seg/partseg1018/checkpoints/best_model.pth',
                        help='checkpoint for pred.seg')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA visible devices (e.g., 0 or 3)')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing outputs')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f'data_root not found: {data_root}')

    ckpt_mae = Path(args.ckpt_mae)
    ckpt_pred = Path(args.ckpt_pred)
    if not ckpt_mae.is_file():
        raise FileNotFoundError(f'ckpt_mae not found: {ckpt_mae}')
    if not ckpt_pred.is_file():
        raise FileNotFoundError(f'ckpt_pred not found: {ckpt_pred}')

    print('Loading models...')
    model_mae = build_and_load_model(ckpt_mae, device)
    model_pred = build_and_load_model(ckpt_pred, device)

    # Iterate sample folders (depth-1)
    folders = [p for p in data_root.iterdir() if p.is_dir()]
    folders.sort()

    for folder in folders:
        # Find a point file
        pts_files = []
        for ext in ('.pts', '.txt', '.xyz', '.npy'):
            pts_files += list(folder.glob(f'*{ext}'))
        if not pts_files:
            print(f'[Skip] No point file in {folder}')
            continue
        pts_path = sorted(pts_files)[0]

        # Determine category
        cat = detect_category_from_folder(folder)
        if cat is None:
            print(f'[Warn] Unknown category for {folder.name}, defaulting to Airplane')
            cat = 'Airplane'
        if cat not in CAT2ID:
            print(f'[Skip] Unrecognized category mapping for {folder.name} -> {cat}')
            continue
        cat_id = CAT2ID[cat]

        out_mae = folder / 'mae.seg'
        out_pred = folder / 'pred.seg'
        if out_mae.exists() and not args.overwrite:
            print(f'[Keep] {out_mae} exists')
        else:
            print(f'[Run ] {folder.name}: {pts_path.name} -> mae.seg ({cat})')
            infer_and_save(model_mae, pts_path, cat_id, out_mae, device)

        if out_pred.exists() and not args.overwrite:
            print(f'[Keep] {out_pred} exists')
        else:
            print(f'[Run ] {folder.name}: {pts_path.name} -> pred.seg ({cat})')
            infer_and_save(model_pred, pts_path, cat_id, out_pred, device)

    print('Done.')


if __name__ == '__main__':
    main()

