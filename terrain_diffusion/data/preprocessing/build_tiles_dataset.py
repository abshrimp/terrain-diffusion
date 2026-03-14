"""Build a training HDF5 dataset from local .npy elevation tiles."""

from pathlib import Path
import random

import click
import h5py
import numpy as np
import skimage.transform
from skimage.measure import block_reduce

from terrain_diffusion.data.laplacian_encoder import laplacian_encode


def _parse_tile_name(path: Path):
    stem = path.stem
    parts = stem.split("_")
    if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
        z, x, y = (int(p) for p in parts)
        return {
            "chunk_id": f"z{z}_x{x}_y{y}",
            "z": z,
            "x": x,
            "y": y,
        }
    return {
        "chunk_id": stem,
        "z": -1,
        "x": -1,
        "y": -1,
    }


@click.command()
@click.option("--tiles-folder", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True, help="Folder containing .npy tiles")
@click.option("--output-file", type=click.Path(path_type=Path), default=Path("data/dataset_tiles.h5"), show_default=True, help="Output HDF5 dataset path")
@click.option("--resolution", type=int, default=30, show_default=True, help="Resolution group label used in HDF5")
@click.option("--split-ratio", type=float, default=0.2, show_default=True, help="Validation split ratio in [0, 1)")
@click.option("--seed", type=int, default=74, show_default=True, help="Random seed for train/val split")
@click.option("--lowres-sigma", type=float, default=5.0, show_default=True, help="Gaussian sigma for low-frequency encoding")
@click.option("--downsample-factor", type=int, default=8, show_default=True, help="Highres to lowres factor (must be >= 1)")
@click.option("--target-size", type=int, default=0, show_default=True, help="Optional square resize for each tile before processing. 0 means no resize")
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it already exists")
def build_tiles_dataset(
    tiles_folder: Path,
    output_file: Path,
    resolution: int,
    split_ratio: float,
    seed: int,
    lowres_sigma: float,
    downsample_factor: int,
    target_size: int,
    overwrite: bool,
):
    """Convert local .npy DEM tiles into the HDF5 schema used by training datasets."""
    if not (0.0 <= split_ratio < 1.0):
        raise click.ClickException("--split-ratio must be in [0, 1)")
    if downsample_factor < 1:
        raise click.ClickException("--downsample-factor must be >= 1")
    if target_size < 0:
        raise click.ClickException("--target-size must be >= 0")

    tile_paths = sorted(tiles_folder.glob("*.npy"))
    if not tile_paths:
        raise click.ClickException(f"No .npy files found in {tiles_folder}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not overwrite:
        raise click.ClickException(f"{output_file} already exists. Use --overwrite to replace it")
    if output_file.exists() and overwrite:
        output_file.unlink()

    rng = random.Random(seed)
    shuffled = tile_paths[:]
    rng.shuffle(shuffled)
    num_val = int(round(len(shuffled) * split_ratio))
    val_set = {p.name for p in shuffled[:num_val]}

    residual_count = 0
    residual_mean = 0.0
    residual_m2 = 0.0
    kept_tiles = 0

    with h5py.File(output_file, "w") as f:
        res_group = f.require_group(str(resolution))

        with click.progressbar(tile_paths, label="Converting tiles") as progress:
            for tile_path in progress:
                tile = np.load(tile_path)
                tile = np.asarray(tile, dtype=np.float32)

                if tile.ndim > 2:
                    tile = np.squeeze(tile)
                if tile.ndim != 2:
                    click.echo(f"Skipping {tile_path.name}: expected 2D array, got shape {tile.shape}")
                    continue

                tile[np.isclose(tile, -9999.0)] = np.nan
                tile[np.isinf(tile)] = np.nan
                if np.isnan(tile).all():
                    click.echo(f"Skipping {tile_path.name}: all values are NaN")
                    continue

                if np.isnan(tile).any():
                    fill_value = np.nanmedian(tile)
                    tile = np.nan_to_num(tile, nan=float(fill_value))

                if target_size > 0 and (tile.shape[0] != target_size or tile.shape[1] != target_size):
                    tile = skimage.transform.resize(
                        tile,
                        (target_size, target_size),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.float32)

                height, width = tile.shape
                cropped_h = (height // downsample_factor) * downsample_factor
                cropped_w = (width // downsample_factor) * downsample_factor
                if cropped_h < downsample_factor or cropped_w < downsample_factor:
                    click.echo(
                        f"Skipping {tile_path.name}: tile too small after factor alignment ({height}x{width})"
                    )
                    continue

                if cropped_h != height or cropped_w != width:
                    tile = tile[:cropped_h, :cropped_w]

                signed_sqrt = np.sign(tile) * np.sqrt(np.abs(tile))
                lowres_shape = (signed_sqrt.shape[0] // downsample_factor, signed_sqrt.shape[1] // downsample_factor)

                residual, lowfreq = laplacian_encode(signed_sqrt, lowres_shape, sigma=lowres_sigma)
                lowres_exact = block_reduce(signed_sqrt, (downsample_factor, downsample_factor), np.median)
                climate = np.repeat(lowres_exact[None, ...], 19, axis=0).astype(np.float32)
                pct_land = float(np.mean(lowfreq > 0))

                meta = _parse_tile_name(tile_path)
                chunk_id = meta["chunk_id"]
                subchunk_id = "chunk_0_0"
                split = "val" if tile_path.name in val_set else "train"

                chunk_group = res_group.require_group(chunk_id)
                if subchunk_id in chunk_group:
                    del chunk_group[subchunk_id]
                subchunk_group = chunk_group.create_group(subchunk_id)
                subchunk_group.attrs["beauty_score"] = 3.0

                for data_type, data in (("residual", residual), ("lowfreq", lowfreq), ("lowres_exact", lowres_exact), ("climate", climate)):
                    dset = subchunk_group.create_dataset(data_type, data=data.astype(np.float32), compression="lzf")
                    dset.attrs.update(
                        {
                            "pct_land": pct_land,
                            "resolution": float(resolution),
                            "data_type": data_type,
                            "chunk_id": chunk_id,
                            "subchunk_id": subchunk_id,
                            "split": split,
                            "source_file": tile_path.name,
                            "z": int(meta["z"]),
                            "x": int(meta["x"]),
                            "y": int(meta["y"]),
                        }
                    )

                flat = residual.reshape(-1)
                for value in flat:
                    residual_count += 1
                    delta = float(value) - residual_mean
                    residual_mean += delta / residual_count
                    delta2 = float(value) - residual_mean
                    residual_m2 += delta * delta2

                kept_tiles += 1

        if kept_tiles == 0:
            raise click.ClickException("No valid tiles were converted")

        residual_std = (residual_m2 / max(residual_count - 1, 1)) ** 0.5
        res_group.attrs["residual_mean"] = float(residual_mean)
        res_group.attrs["residual_std"] = float(residual_std)
        res_group.attrs["source"] = "local_npy_tiles"
        res_group.attrs["downsample_factor"] = int(downsample_factor)
        res_group.attrs["lowres_sigma"] = float(lowres_sigma)

    click.echo(f"Created {output_file} with {kept_tiles} tiles (resolution group: {resolution})")
    click.echo(f"Residual stats: mean={residual_mean:.6f}, std={residual_std:.6f}")


if __name__ == "__main__":
    build_tiles_dataset()