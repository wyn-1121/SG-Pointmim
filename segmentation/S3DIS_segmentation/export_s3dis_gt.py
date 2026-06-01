"""
Export S3DIS ground-truth labels (no model inference).

For rooms in a given Area, saves:
- <room>.gt.txt  (per-point labels 0..12)
- <room>.gt.ply  (optional, colored point cloud by GT labels)

Usage examples:
  # Export one room (txt + ply)
  python segmentation/export_s3dis_gt.py \
    --root ../data/stanford_indoor3d/ \
    --test_area 5 \
    --room Area_5_office_1 \
    --save_dir ./segmentation/preds_s3dis \
    --export_ply

  # Export all rooms in Area 5 (txt only)
  python segmentation/export_s3dis_gt.py \
    --root ../data/stanford_indoor3d/ \
    --test_area 5 \
    --save_dir ./segmentation/preds_s3dis
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
from typing import Optional

SEG_DIR = Path(__file__).parent
sys.path.append(str(SEG_DIR))
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene  # type: ignore


# Fixed color palette for 13 S3DIS classes (default)
S3DIS_COLORS_DEFAULT = np.array([
    [  0, 255, 255],  # ceiling - cyan
    [255,   0, 255],  # floor   - magenta
    [255, 255,   0],  # wall    - yellow
    [255, 128,   0],  # beam    - orange
    [128,   0, 255],  # column  - purple
    [  0, 128, 255],  # window  - sky blue
    [  0, 255,   0],  # door    - green
    [128, 128, 128],  # table   - gray
    [255,   0,   0],  # chair   - red
    [128,  64,   0],  # sofa    - brown
    [  0,  64, 128],  # bookcase- dark blue
    [ 64, 128,   0],  # board   - olive
    [  0,   0,   0],  # clutter - black
], dtype=np.uint8)


def load_palette(palette: str, palette_file: Optional[str]) -> np.ndarray:
    if palette_file:
        arr = []
        with open(palette_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # support comma or whitespace
                parts = line.replace(',', ' ').split()
                if len(parts) < 3:
                    continue
                r, g, b = map(int, parts[:3])
                arr.append([r, g, b])
        arr = np.array(arr, dtype=np.uint8)
        if arr.shape[0] < 13:
            # pad with zeros if fewer provided
            pad = np.zeros((13 - arr.shape[0], 3), dtype=np.uint8)
            arr = np.vstack([arr, pad])
        return arr[:13]
    if palette == 'default':
        return S3DIS_COLORS_DEFAULT
    if palette == 'random':
        rng = np.random.RandomState(42)
        return rng.randint(0, 256, size=(13, 3), dtype=np.uint8)
    # fallback
    return S3DIS_COLORS_DEFAULT


def _sanitize_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray):
    n0 = xyz.shape[0]
    mask = np.isfinite(xyz).all(axis=1)
    if mask.sum() != n0:
        xyz = xyz[mask]
        rgb = rgb[mask]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return xyz.astype(np.float32), rgb


def write_ply_xyzrgb(xyz: np.ndarray, rgb: np.ndarray, out_path: Path, fmt: str = 'ascii'):
    xyz, rgb = _sanitize_xyz_rgb(xyz, rgb)
    n = xyz.shape[0]
    if fmt == 'binary':
        import struct
        with open(out_path, 'wb') as f:
            header = (
                'ply\n'
                'format binary_little_endian 1.0\n'
                f'element vertex {n}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n'
                'property uchar red\n'
                'property uchar green\n'
                'property uchar blue\n'
                'end_header\n'
            )
            f.write(header.encode('ascii'))
            for i in range(n):
                f.write(struct.pack('<fffBBB', float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]), int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])))
    else:
        with open(out_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {n}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(n):
                x, y, z = xyz[i]
                r, g, b = rgb[i]
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n')


def main():
    parser = argparse.ArgumentParser(description='Export S3DIS ground-truth labels (whole-scene)')
    parser.add_argument('--root', type=str, default='../data/stanford_indoor3d/', help='S3DIS data root')
    parser.add_argument('--test_area', type=int, default=5, help='Area id to export [1..6]')
    parser.add_argument('--block_points', type=int, default=4096, help='points per block (unused; for loader compatibility)')
    parser.add_argument('--block_size', type=float, default=1.0, help='block size in meters (unused here)')
    parser.add_argument('--stride', type=float, default=0.5, help='sliding stride (unused here)')
    parser.add_argument('--save_dir', type=str, default=str(SEG_DIR / 'preds_s3dis'), help='output folder')
    parser.add_argument('--room', type=str, default=None, help='room file name or stem to export only that room')
    parser.add_argument('--export_ply', action='store_true', help='also export a colored PLY per room using palette')
    parser.add_argument('--ply_format', type=str, default='ascii', choices=['ascii', 'binary'], help='PLY output format')
    parser.add_argument('--palette', type=str, default='default', choices=['default', 'random'], help='which palette to use if not providing palette_file')
    parser.add_argument('--palette_file', type=str, default=None, help='custom palette file (13 lines of r g b or r,g,b)')
    parser.add_argument('--index_base', type=int, default=0, help='if labels are 1-based in your data, set to 1 to shift to 0..12')
    args = parser.parse_args()

    root = args.root
    if not root.endswith(os.sep):
        root = root + os.sep

    dataset = ScannetDatasetWholeScene(
        root=root,
        block_points=args.block_points,
        split='test',
        test_area=args.test_area,
        stride=args.stride,
        block_size=args.block_size,
        padding=0.001,
    )

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(dataset)))
    if args.room is not None:
        target = args.room
        matches = []
        for i, fname in enumerate(dataset.file_list):
            stem = Path(fname).stem
            if fname == target or stem == target or target in fname:
                matches.append(i)
        if not matches:
            raise FileNotFoundError(f'Room "{target}" not found in Area_{args.test_area} under {root}. Example: {dataset.file_list[:3]}')
        indices = matches

    # load palette
    palette_arr = load_palette(args.palette, args.palette_file)

    for room_idx in indices:
        room_file = dataset.file_list[room_idx]
        xyzrgb = dataset.scene_points_list[room_idx]  # [N,6]
        xyz = xyzrgb[:, :3]
        gt_raw = dataset.semantic_labels_list[room_idx].astype(np.int32)  # [N]
        # shift if labels are 1-based
        gt = gt_raw - args.index_base

        gt_txt = save_root / (Path(room_file).stem + '.gt.txt')
        np.savetxt(gt_txt, gt, fmt='%d')
        print(f'Saved {gt_txt} (N={gt.shape[0]})')

        if args.export_ply:
            color = palette_arr[np.clip(gt, 0, 12)]
            ply_path = save_root / (Path(room_file).stem + '.gt.ply')
            write_ply_xyzrgb(xyz, color, ply_path, fmt=args.ply_format)
            print(f'Saved {ply_path} (colored GT)')


if __name__ == '__main__':
    main()
