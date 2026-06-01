"""
Single-shape inference for ShapeNet Part segmentation.

- Loads a trained checkpoint for `segmentation/models/pt.py` (default).
- Accepts an input point cloud file (`.txt`, `.pts`, `.xyz`, `.npy`).
- If you only have a `.seg` file path, the script will try to find the
  paired points file with the same stem and one of the known extensions.
- Requires a category (16 classes in ShapeNetPart). You can specify with
  `--category` (e.g., Chair) or `--cat_id` (0..15). If you provided a ground
  truth `.seg` file with the same stem, the script will infer the category
  from the majority label in that `.seg` using the standard label mapping.

Outputs a `<stem>_pred.seg` text file with one integer label per point
in the same order as the input points fed to the model.

Usage examples:

  python segmentation/infer_partseg.py \
    --ckpt path/to/checkpoint.pth \
    --input /path/to/shape.txt \
    --category Chair

  # If you only have a .seg path from the raw dataset layout:
  python segmentation/infer_partseg.py \
    --ckpt path/to/checkpoint.pth \
    --input /path/to/873f4...e0c.seg \
    --try_pair  # tries to locate paired .pts/.txt/.xyz with same stem
"""

import argparse
import os
import sys
import numpy as np
import torch
import importlib
from pathlib import Path

# Local imports follow repository structure
SYS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SYS_DIR, 'models'))

from pointnet_util import pc_normalize  # segmentation/pointnet_util.py


# Mapping from part labels (0..49) to categories
SEG_CLASSES = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
    'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
    'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
}


from typing import Optional


def build_cat_mappings(shapenet_root: Optional[str]):
    """Build mappings:
    - cat_name_to_id: e.g., 'Chair' -> int id in [0..15]
    - id_to_cat_name: reverse mapping

    If shapenet_root provided, respect category order in
    `synsetoffset2category.txt`. Otherwise, fallback to the canonical
    ShapeNetPart order:
    Airplane, Bag, Cap, Car, Chair, Earphone, Guitar, Knife,
    Lamp, Laptop, Motorbike, Mug, Pistol, Rocket, Skateboard, Table.
    """
    cat_names = [
        'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
        'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'
    ]
    if shapenet_root:
        syn_path = Path(shapenet_root) / 'synsetoffset2category.txt'
        if syn_path.is_file():
            ordered = []
            with open(syn_path, 'r') as f:
                for line in f:
                    name = line.strip().split()[0]
                    if name in SEG_CLASSES:
                        ordered.append(name)
            if len(ordered) == len(SEG_CLASSES):
                cat_names = ordered
    # default fallback order
    cat_name_to_id = {name: idx for idx, name in enumerate(cat_names)}
    id_to_cat_name = {v: k for k, v in cat_name_to_id.items()}
    return cat_name_to_id, id_to_cat_name


def try_find_points_for_seg(seg_path: Path) -> Optional[Path]:
    stem = seg_path.with_suffix('')
    candidates = [stem.with_suffix(ext) for ext in ('.txt', '.pts', '.xyz', '.npy')]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_points(path: Path) -> np.ndarray:
    """Load points from supported formats; returns (N,3) in float32.
    - .txt: whitespace or comma separated; take first 3 columns
    - .pts/.xyz: whitespace separated; first 3 columns
    - .npy: Nx3 array
    """
    if path.suffix.lower() == '.npy':
        arr = np.load(path)
        pts = arr[:, :3].astype(np.float32)
    else:
        # try both comma and whitespace
        try:
            data = np.loadtxt(path, delimiter=',').astype(np.float32)
        except ValueError:
            data = np.loadtxt(path).astype(np.float32)
        if data.ndim == 1:
            data = data[None, :]
        pts = data[:, :3].astype(np.float32)
    return pts


def to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    eye = torch.eye(num_classes, device=y.device)
    return eye[y.view(-1)]


def infer_once(model, points_xyz: np.ndarray, cat_id: int, device: torch.device,
               id_to_cat_name: dict) -> np.ndarray:
    # normalize like training
    pts_norm = pc_normalize(points_xyz.copy())
    pts = torch.from_numpy(pts_norm).unsqueeze(0).float().to(device)  # [1, N, 3]
    pts = pts.transpose(2, 1)  # [1, 3, N]
    label = torch.tensor([cat_id], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(pts, to_categorical(label, 16))  # [1, N, 50]
        # Restrict prediction to the valid part labels of the given category
        cat_name = id_to_cat_name[int(cat_id)]
        valid = SEG_CLASSES[cat_name]
        # logits: [1, N, 50] -> select columns
        selected = logits[..., valid]
        local_idx = selected.argmax(dim=-1)  # [1, N]
        pred = (local_idx + valid[0]).squeeze(0).detach().cpu().numpy()  # map back to global label id
    return pred


def majority_cat_from_seg(seg_file: Path) -> Optional[str]:
    try:
        seg = np.loadtxt(seg_file).astype(np.int32)
        if seg.ndim == 2:
            seg = seg[:, -1]
        # pick the category whose parts contain the majority label
        values, counts = np.unique(seg, return_counts=True)
        top_label = int(values[np.argmax(counts)])
        for cat, labels in SEG_CLASSES.items():
            if top_label in labels:
                return cat
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Infer part segmentation for a single shape')
    parser.add_argument('--model', type=str, default='pt', help='model name under segmentation/models (default: pt)')
    parser.add_argument('--ckpt', type=str, required=True, help='path to trained checkpoint .pth')
    parser.add_argument('--input', type=str, required=True, help='path to input point cloud or .seg file')
    parser.add_argument('--output', type=str, default=None, help='output .seg path (default: <stem>_pred.seg)')
    parser.add_argument('--category', type=str, default=None, help='category name (e.g., Chair)')
    parser.add_argument('--cat_id', type=int, default=None, help='category id [0..15] if known')
    parser.add_argument('--shapenet_root', type=str, default=None, help='dataset root to resolve cat id ordering')
    parser.add_argument('--try_pair', action='store_true', help='when input is .seg, try to find paired points file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use (default: 0)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cat_name_to_id, id_to_cat_name = build_cat_mappings(args.shapenet_root)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f'Input not found: {in_path}')

    # Resolve points file and possibly infer category
    points_path = in_path
    inferred_cat = None
    if in_path.suffix.lower() == '.seg':
        if args.try_pair:
            paired = try_find_points_for_seg(in_path)
            if paired is None:
                raise FileNotFoundError(f'Cannot find paired points for {in_path} (.txt/.pts/.xyz/.npy)')
            points_path = paired
        else:
            raise ValueError('Input is a .seg label file. Provide the paired points file via --input, or use --try_pair.')
        # try to infer category from GT seg file
        inferred_cat = majority_cat_from_seg(in_path)

    pts = load_points(points_path)

    # Determine category id
    cat_id = None
    if args.cat_id is not None:
        cat_id = int(args.cat_id)
    elif args.category is not None:
        if args.category not in cat_name_to_id:
            valid = ', '.join(sorted(cat_name_to_id.keys()))
            raise ValueError(f'Unknown category {args.category}. Valid: {valid}')
        cat_id = cat_name_to_id[args.category]
    elif inferred_cat is not None:
        cat_id = cat_name_to_id[inferred_cat]
    else:
        valid = ', '.join(sorted(cat_name_to_id.keys()))
        raise ValueError(f'Category required. Use --category/--cat_id, or pass a paired ground-truth .seg to infer. Valid: {valid}')

    # Load model
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(cls_dim=50).to(device)
    model.eval()

    # Robust checkpoint loading: prefer segmentation checkpoints with 'model_state_dict'.
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        model.load_state_dict(state, strict=False)
    elif isinstance(ckpt, dict) and 'base_model' in ckpt:
        # pretrain-style ckpt; mimic load_model_from_ckpt without hardcoded cuda index
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('MAE_encoder'):
                base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]
        model.load_state_dict(base_ckpt, strict=False)
    else:
        # Raw state_dict
        model.load_state_dict(ckpt, strict=False)

    pred = infer_once(model, pts, cat_id, device, id_to_cat_name)

    # Save output
    out_path = Path(args.output) if args.output else points_path.with_suffix('').with_name(points_path.stem + '_pred.seg')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, pred.astype(np.int32), fmt='%d')
    cat_name = id_to_cat_name.get(cat_id, str(cat_id))
    print(f'Saved prediction to {out_path} (category: {cat_name}, points: {len(pred)})')


if __name__ == '__main__':
    main()
