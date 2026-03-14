#!/usr/bin/env python3
"""Generate high-resolution terrain detail from a saved coarse world.

Takes the output directory produced by generate_coarse.py (which contains
world.h5 + params.json) and generates a combined high-resolution elevation
image for the specified coarse coordinate range.

Internally uses the same base+decoder pipeline as the interactive explorer:
  - Each coarse pixel maps to 256 native pixels.
  - The base model generates 64-pixel latent tiles (stride 32).
  - The decoder upscales latents 8× to native resolution.

Usage examples
--------------
# Generate a 4×4 coarse-pixel detail region (centred around origin)
python util_scripts/generate_detail.py \\
    ../terrain-diffusion-30m outputs/coarse_worlds/seed_12345 \\
    --ci0 -2 --ci1 2 --cj0 -2 --cj1 2

# Larger region, explicit output directory, higher batch size
python util_scripts/generate_detail.py \\
    ../terrain-diffusion-30m outputs/coarse_worlds/seed_12345 \\
    --ci0 -5 --ci1 5 --cj0 -5 --cj1 5 \\
    --output-dir outputs/detail_region --batch-size 4 --log-mode verbose
"""
import json
import os
import sys

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running directly from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.common.cli_helpers import parse_kwargs

# Native pixels per coarse pixel (matches world_pipeline internals)
COARSE_TO_NATIVE = 256


@click.command()
@click.argument("model_path")
@click.argument("world_dir")
@click.option("--ci0", type=int, required=True,
              help="Start coarse i index (inclusive). Visible in coarse_elev.png Y-axis.")
@click.option("--ci1", type=int, required=True,
              help="End coarse i index (exclusive).")
@click.option("--cj0", type=int, required=True,
              help="Start coarse j index (inclusive). Visible in coarse_elev.png X-axis.")
@click.option("--cj1", type=int, required=True,
              help="End coarse j index (exclusive).")
@click.option("--output-dir", default=None,
              help="Output directory. Default: <world_dir>/detail_ci<ci0>_<ci1>_cj<cj0>_<cj1>.")
@click.option("--batch-size", "batch_size_str", type=str, default="1,4", show_default=True,
              help="Batch size(s) for latent generation, e.g. '4' or '1,2,4,8'.")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose",
              show_default=True, help="Logging verbosity.")
@click.option("--compile/--no-compile", "torch_compile", default=False, show_default=True,
              help="Use torch.compile for faster inference.")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32",
              show_default=True)
@click.option("--device", default=None,
              help="Compute device (cuda/cpu). Default: auto-detect.")
@click.option("--save-npy/--no-save-npy", default=True, show_default=True,
              help="Save elevation as a float32 .npy file.")
@click.option("--save-png/--no-save-png", default=True, show_default=True,
              help="Save relief and elevation colormap PNG images.")
@click.option("--hydro-enforce/--no-hydro-enforce", default=False, show_default=True,
              help="Apply hydrology consistency (depression fill + optional smoothing).")
@click.option("--hydro-max-raise", type=float, default=120.0, show_default=True,
              help="Maximum depression fill depth in meters. Use negative to disable the cap.")
@click.option("--hydro-epsilon", type=float, default=1e-3, show_default=True,
              help="Small gradient added when filling flats.")
@click.option("--hydro-connectivity", type=click.Choice(["4", "8"]), default="8", show_default=True,
              help="Neighbor connectivity for depression filling.")
@click.option("--hydro-smooth-iters", type=int, default=0, show_default=True,
              help="Optional river bump smoothing iterations after fill.")
@click.option("--kwarg", "extra_kwargs", multiple=True,
              help="Extra key=value pipeline kwargs.")
def main(
    model_path,
    world_dir,
    ci0,
    ci1,
    cj0,
    cj1,
    output_dir,
    batch_size_str,
    log_mode,
    torch_compile,
    dtype,
    device,
    save_npy,
    save_png,
    hydro_enforce,
    hydro_max_raise,
    hydro_epsilon,
    hydro_connectivity,
    hydro_smooth_iters,
    extra_kwargs,
):
    """Generate a high-resolution elevation tile from a saved coarse world.

    MODEL_PATH is the same local directory or HuggingFace Hub model ID used
    when running generate_coarse.py.

    WORLD_DIR is the per-seed output directory produced by generate_coarse.py
    (it must contain world.h5 and params.json).

    The coarse coordinate range [CI0,CI1) × [CJ0,CJ1) selects which portion
    of the coarse map to upscale.  The i/j axes match the Y/X axes displayed
    in coarse_elev.png.  1 coarse pixel = 256 native pixels (≈23 km at 90 m/px).

    Outputs
    -------
    elevation.npy   – float32 (H, W) elevation array in metres
    relief.png      – shaded relief image
    elevation.png   – terrain-colourmap PNG
    result_params.json – bounding box and summary statistics
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if ci1 <= ci0:
        raise click.BadParameter(f"ci1 ({ci1}) must be greater than ci0 ({ci0})")
    if cj1 <= cj0:
        raise click.BadParameter(f"cj1 ({cj1}) must be greater than cj0 ({cj0})")

    hdf5_path = os.path.join(world_dir, 'world.h5')
    if not os.path.exists(hdf5_path):
        raise click.ClickException(
            f"HDF5 cache not found: {hdf5_path}\n"
            "Run generate_coarse.py first to create the world cache."
        )

    # ── Output directory ──────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(
            world_dir, f"detail_ci{ci0}_{ci1}_cj{cj0}_{cj1}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # ── Parse batch sizes ─────────────────────────────────────────────────────
    if ',' in batch_size_str:
        batch_sizes = [int(x.strip()) for x in batch_size_str.split(',')]
    else:
        batch_sizes = int(batch_size_str)

    # ── Normalise dtype ───────────────────────────────────────────────────────
    if dtype == 'fp32':
        dtype = None

    # ── Device selection ──────────────────────────────────────────────────────
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")

    extra = parse_kwargs(extra_kwargs)
    hydro_max_raise_effective = None if hydro_max_raise < 0 else hydro_max_raise

    # ── Read seed from params.json ────────────────────────────────────────────
    seed = None
    params_path = os.path.join(world_dir, 'params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            saved_params = json.load(f)
        seed = saved_params.get('seed')
        print(f"Loaded seed from params.json: {seed}")
    else:
        print("Warning: params.json not found; seed will be read from HDF5 metadata.")

    # ── Build pipeline (seed reconciled from HDF5 if needed) ─────────────────
    world = WorldPipeline.from_pretrained(
        model_path,
        seed=seed,
        latents_batch_size=batch_sizes,
        log_mode=log_mode,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy='indirect',
        hydrology_enforce=hydro_enforce,
        hydrology_max_raise=hydro_max_raise_effective,
        hydrology_epsilon=hydro_epsilon,
        hydrology_connectivity=int(hydro_connectivity),
        hydrology_smooth_iterations=hydro_smooth_iters,
        **extra,
    )
    world.to(device)
    world.bind(hdf5_path)  # will reconcile seed with HDF5 if there is a mismatch

    with world:
        print(f"World seed: {world.seed}")

        # ── Map coarse → native coords ────────────────────────────────────────
        i1 = ci0 * COARSE_TO_NATIVE
        i2 = ci1 * COARSE_TO_NATIVE
        j1 = cj0 * COARSE_TO_NATIVE
        j2 = cj1 * COARSE_TO_NATIVE
        h = i2 - i1
        w = j2 - j1
        res = world.native_resolution

        print(f"Detail region  : coarse i=[{ci0},{ci1}), j=[{cj0},{cj1})")
        print(f"Native coords  : i=[{i1},{i2}), j=[{j1},{j2})")
        print(f"Output size    : {h} × {w} px  "
              f"({h * res / 1000:.1f} km × {w * res / 1000:.1f} km "
              f"at {res:.0f} m/px)")

        # ── Generate elevation ────────────────────────────────────────────────
        region = world.get(i1, j1, i2, j2, with_climate=False)
        elev = region['elev']
        elev_np = elev.detach().cpu().float().numpy()

        elev_min = float(np.nanmin(elev_np))
        elev_max = float(np.nanmax(elev_np))
        elev_mean = float(np.nanmean(elev_np))
        print(f"Elevation      : min={elev_min:.1f} m, "
              f"max={elev_max:.1f} m, mean={elev_mean:.1f} m")

        # ── Save numpy elevation ──────────────────────────────────────────────
        if save_npy:
            npy_path = os.path.join(output_dir, 'elevation.npy')
            np.save(npy_path, elev_np)
            print(f"Saved elevation data  : {npy_path}")

        # ── Save PNG outputs ──────────────────────────────────────────────────
        if save_png:
            # Shaded relief
            relief_rgb = get_relief_map(elev_np, None, None, None,
                                        resolution=res)
            relief_path = os.path.join(output_dir, 'relief.png')
            plt.imsave(relief_path, np.clip(relief_rgb, 0.0, 1.0))
            print(f"Saved relief map      : {relief_path}")

            # Terrain-colormap elevation
            fig, ax = plt.subplots(figsize=(12, 12))
            im = ax.imshow(
                elev_np,
                cmap='terrain',
                origin='lower',
                extent=[j1, j2, i1, i2],
            )
            plt.colorbar(im, ax=ax, label='Elevation (m)')
            ax.set_title(
                f'Elevation  |  seed={world.seed}  |  '
                f'ci=[{ci0},{ci1}), cj=[{cj0},{cj1})'
            )
            ax.set_xlabel('j  (native pixel)')
            ax.set_ylabel('i  (native pixel)')
            elev_png_path = os.path.join(output_dir, 'elevation.png')
            fig.savefig(elev_png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved elevation map   : {elev_png_path}")

        # ── result_params.json ────────────────────────────────────────────────
        result = {
            'seed': world.seed,
            'model_path': model_path,
            'ci0': ci0, 'ci1': ci1,
            'cj0': cj0, 'cj1': cj1,
            'i1': i1, 'i2': i2,
            'j1': j1, 'j2': j2,
            'height_px': h,
            'width_px': w,
            'native_resolution_m': res,
            'hydrology_enforce': hydro_enforce,
            'hydrology_max_raise_m': hydro_max_raise_effective,
            'hydrology_epsilon': hydro_epsilon,
            'hydrology_connectivity': int(hydro_connectivity),
            'hydrology_smooth_iterations': hydro_smooth_iters,
            'elev_min_m': elev_min,
            'elev_max_m': elev_max,
            'elev_mean_m': elev_mean,
        }
        result_path = os.path.join(output_dir, 'result_params.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved result params   : {result_path}")

        print(f"\nAll outputs saved to  : {output_dir}")


if __name__ == '__main__':
    main()
