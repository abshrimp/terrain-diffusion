#!/usr/bin/env python3
"""
日本の地形タイル (data/custom_tiles/12_{x}_{y}.npy) を
terrain-diffusion 学習用 HDF5 データセットに変換するスクリプト。

処理内容:
  1. 隣接する 2x2 タイル (各256x256) を 512x512 チャンクに結合
  2. 標高を [0, 3200m] にクリップ (-9999 / 負値 → 0m)
  3. signed-sqrt 変換: sqrt(x) を適用 → [0, 56.57]
  4. ラプラシアン分解: residual (512x512) + lowfreq (64x64)
  5. HDF5 に保存

使い方:
  cd terrain-diffusion
  python data/build_japan_dataset.py \
      --tile-dir data/custom_tiles \
      --output data/japan_dataset.h5
"""

import sys
import click
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm

# terrain_diffusion パッケージをインポートできるようにルートを追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrain_diffusion.data.laplacian_encoder import laplacian_encode

# ---- 定数 ----------------------------------------------------------------
ZOOM = 12
TILE_SIZE = 256        # 各タイルのピクセルサイズ
MAX_ELEVATION = 3200.0 # 最大標高 [m]
NODATA_THRESHOLD = -9990.0  # この値より小さければ nodata (海)
SIGMA = 5.0            # ガウスぼかしの sigma
DOWNSAMPLE_FACTOR = 8  # lowfreq は 1/8 サイズ
RES_GROUP = "90"       # HDF5 内の解像度グループ名 (オリジナルに合わせる)
# ---------------------------------------------------------------------------


def signed_sqrt(x: np.ndarray) -> np.ndarray:
    """Signed square-root 変換: sign(x) * sqrt(|x|)  (x>=0 なら単に sqrt(x))"""
    return np.sign(x) * np.sqrt(np.abs(x))


def load_tile(tile_dir: Path, x: int, y: int):
    """1 枚のタイルを読み込む。存在しない場合はゼロ埋め (海) を返す。"""
    path = tile_dir / f"{ZOOM}_{x}_{y}.npy"
    if path.exists():
        data = np.load(str(path)).astype(np.float32)
        land_mask = data > 0          # 正の標高 = 陸地
        # nodata (-9999) と負値 → 0m、上限 3200m
        data = np.where(data < NODATA_THRESHOLD, 0.0, data)
        data = np.clip(data, 0.0, MAX_ELEVATION)
        return data, land_mask
    return (
        np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32),
        np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool),
    )


def find_tile_groups(tile_dir: Path, group_n: int):
    """全タイルを探索して NxN のグループに分類する。"""
    tile_coords: set = set()
    for f in tile_dir.glob(f"{ZOOM}_*.npy"):
        parts = f.stem.split("_")
        if len(parts) == 3 and parts[0] == str(ZOOM):
            try:
                tile_coords.add((int(parts[1]), int(parts[2])))
            except ValueError:
                pass

    groups: dict = {}
    for (x, y) in tile_coords:
        gx = (x // group_n) * group_n
        gy = (y // group_n) * group_n
        groups.setdefault((gx, gy), set()).add((x, y))

    return sorted(groups.keys()), tile_coords


def build_chunk(tile_dir: Path, gx: int, gy: int, group_n: int):
    """NxN タイルを結合してひとつのチャンクにする。"""
    n = group_n * TILE_SIZE
    chunk = np.zeros((n, n), dtype=np.float32)
    land_mask = np.zeros((n, n), dtype=bool)

    for dx in range(group_n):
        for dy in range(group_n):
            tile, mask = load_tile(tile_dir, gx + dx, gy + dy)
            r0 = dx * TILE_SIZE
            c0 = dy * TILE_SIZE
            chunk[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile
            land_mask[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = mask

    return chunk, land_mask


def compute_beauty_score(chunk_sqrt: np.ndarray) -> float:
    """勾配の大きさに基づく簡易的な景観スコア (1〜5)。"""
    grad_y = np.gradient(chunk_sqrt, axis=0)
    grad_x = np.gradient(chunk_sqrt, axis=1)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    score = float(np.clip(np.mean(grad_mag) * 2.0, 1.0, 5.0))
    return score


@click.command()
@click.option("--tile-dir", default="data/custom_tiles", show_default=True,
              help="12_{x}_{y}.npy が入ったディレクトリ")
@click.option("--output", default="data/japan_dataset.h5", show_default=True,
              help="出力 HDF5 ファイルのパス")
@click.option("--group-n", default=2, show_default=True,
              help="NxN タイルをひとつのチャンクに結合 (2→512x512, 4→1024x1024)")
@click.option("--train-frac", default=0.8, show_default=True,
              help="学習用データの割合 (残りは val)")
@click.option("--min-land-pct", default=0.0, show_default=True,
              help="陸地面積がこの割合未満のチャンクをスキップ")
def main(tile_dir: str, output: str, group_n: int,
         train_frac: float, min_land_pct: float):
    """日本地形タイルを terrain-diffusion 用 HDF5 に変換する。"""
    tile_dir_path = Path(tile_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = group_n * TILE_SIZE
    lowfreq_size = chunk_size // DOWNSAMPLE_FACTOR

    print(f"タイルディレクトリ : {tile_dir_path.resolve()}")
    print(f"出力ファイル       : {output_path.resolve()}")
    print(f"グループサイズ     : {group_n}x{group_n} → {chunk_size}x{chunk_size} px")
    print(f"lowfreq サイズ     : {lowfreq_size}x{lowfreq_size} px")

    group_keys, all_tiles = find_tile_groups(tile_dir_path, group_n)
    print(f"タイル数           : {len(all_tiles)}")
    print(f"グループ数         : {len(group_keys)}")

    if len(group_keys) == 0:
        print("ERROR: タイルが見つかりません。--tile-dir を確認してください。")
        raise SystemExit(1)

    n_train = max(1, int(len(group_keys) * train_frac))
    residual_train_vals: list = []
    total_land = 0
    total_ocean = 0
    skipped = 0

    with h5py.File(str(output_path), "w") as f:
        res_grp = f.require_group(RES_GROUP)

        for i, (gx, gy) in enumerate(tqdm(group_keys, desc="チャンク処理中")):
            split = "train" if i < n_train else "val"

            chunk, land_mask = build_chunk(tile_dir_path, gx, gy, group_n)
            pct_land = float(np.mean(land_mask))

            if pct_land < min_land_pct:
                skipped += 1
                continue

            # signed-sqrt 変換 (x >= 0 なので sqrt(x) と等価)
            chunk_sqrt = signed_sqrt(chunk)

            # ラプラシアン分解
            t = torch.from_numpy(chunk_sqrt[None, None])  # [1, 1, H, W]
            residual_t, lowfreq_t = laplacian_encode(t, lowfreq_size, SIGMA)
            residual = residual_t[0, 0].numpy()           # [H, W]
            lowfreq = lowfreq_t[0, 0].numpy()             # [lowfreq_size, lowfreq_size]

            # 景観スコア
            beauty_score = compute_beauty_score(chunk_sqrt)

            # HDF5 に保存
            chunk_id = f"g{gx}_{gy}"
            subchunk_id = "0"
            attrs = {
                "pct_land": pct_land,
                "split": split,
                "chunk_id": chunk_id,
                "subchunk_id": subchunk_id,
            }

            sg = res_grp.require_group(chunk_id).require_group(subchunk_id)
            sg.attrs["beauty_score"] = beauty_score

            # residual と lowfreq を保存
            for name, data, cshape in [
                ("residual",     residual, (min(128, chunk_size), min(128, chunk_size))),
                ("lowfreq",      lowfreq,  (min(16, lowfreq_size), min(16, lowfreq_size))),
                ("lowres_exact", lowfreq,  (min(16, lowfreq_size), min(16, lowfreq_size))),
            ]:
                ds = sg.create_dataset(name, data=data, compression="lzf", chunks=cshape)
                ds.attrs.update(attrs)

            # 学習データの residual 統計収集 (陸地ピクセルのみ)
            if split == "train" and pct_land > 0.01:
                # ダウンサンプル: 1/8 解像度の land_mask で近似
                lm_small = land_mask[::8, ::8]  # roughly same spatial position
                rvals = residual[lm_small].ravel() if lm_small.any() else residual.ravel()
                residual_train_vals.append(rvals[:min(len(rvals), 10000)])

            total_land += int(np.sum(land_mask))
            total_ocean += int(np.sum(~land_mask))

    # ---- 統計出力 --------------------------------------------------------
    if residual_train_vals:
        all_res = np.concatenate(residual_train_vals)
        rmean = float(np.mean(all_res))
        rstd = float(np.std(all_res))
    else:
        rmean, rstd = 0.0, 1.0

    print()
    print("=" * 60)
    print("データセット統計")
    print("=" * 60)
    print(f"総グループ数   : {len(group_keys)}")
    print(f"  train        : {n_train}")
    print(f"  val          : {len(group_keys) - n_train}")
    print(f"  スキップ     : {skipped}")
    print(f"陸地ピクセル   : {total_land:,}")
    print(f"海洋ピクセル   : {total_ocean:,}")
    print(f"residual_mean  : {rmean:.6f}")
    print(f"residual_std   : {rstd:.6f}")
    print()
    print("以下の値を configs/*.cfg に設定してください:")
    print(f"  residual_mean = {rmean:.4f}")
    print(f"  residual_std  = {rstd:.4f}")
    print("=" * 60)
    print(f"\nデータセット保存完了: {output_path}")


if __name__ == "__main__":
    main()
