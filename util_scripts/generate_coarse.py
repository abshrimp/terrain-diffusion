#!/usr/bin/env python3
"""Generate coarse terrain maps from the WorldPipeline.

Generates one or more seeds and saves:
  - coarse_elev.png   : elevation preview image (viridis colormap)
  - coarse_elev.npy   : raw elevation data in metres
  - world.h5          : HDF5 cache (pass to generate_detail.py)
  - params.json       : seed and window parameters

Usage examples
--------------
# Generate 3 random seeds into outputs/coarse_worlds/
python util_scripts/generate_coarse.py ../terrain-diffusion-30m --count 3

# Generate an explicit seed
python util_scripts/generate_coarse.py ../terrain-diffusion-30m --seed 12345

# Shift the coarse window and use a different output directory
python util_scripts/generate_coarse.py ../terrain-diffusion-30m \\
    --count 2 --coarse-window 40 --coarse-offset-i 10 --coarse-offset-j -5 \\
    --output-dir outputs/my_worlds
"""
import json
import os
import random
import sys

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running directly from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from terrain_diffusion.inference.world_pipeline import WorldPipeline, normalize_tensor
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.common.cli_helpers import parse_kwargs


def generate_coarse(
    model_path: str,
    seed: int,
    output_dir: str,
    coarse_window: int,
    coarse_offset_i: int,
    coarse_offset_j: int,
    device: str | None,
    batch_sizes,
    log_mode: str,
    torch_compile: bool,
    dtype: str | None,
    **kwargs,
) -> None:
    """Generate and save the coarse map for a single seed."""
    os.makedirs(output_dir, exist_ok=True)
    hdf5_path = os.path.join(output_dir, 'world.h5')

    world = WorldPipeline.from_pretrained(
        model_path,
        seed=seed,
        latents_batch_size=batch_sizes,
        log_mode=log_mode,
        torch_compile=torch_compile,
        dtype=dtype,
        caching_strategy='indirect',
        **kwargs,
    )
    world.to(device)
    world.bind(hdf5_path)

    with world:
        print(f"World seed: {world.seed}")

        ci0 = coarse_offset_i - coarse_window
        ci1 = coarse_offset_i + coarse_window
        cj0 = coarse_offset_j - coarse_window
        cj1 = coarse_offset_j + coarse_window

        print(f"Generating coarse map: i=[{ci0},{ci1}), j=[{cj0},{cj1}) ...")

        coarse_raw = world.coarse[:, ci0:ci1, cj0:cj1]
        coarse_elev_ss = normalize_tensor(coarse_raw, dim=0)[0]
        coarse_elev_m = torch.sign(coarse_elev_ss) * torch.square(coarse_elev_ss)
        coarse_np = coarse_elev_m.detach().cpu().numpy()

        elev_min = float(np.nanmin(coarse_np))
        elev_max = float(np.nanmax(coarse_np))

        # ── Elevation preview PNG (viridis) ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(
            coarse_np,
            origin='lower',
            interpolation='nearest',
            extent=[cj0, cj1, ci0, ci1],
            cmap='viridis',
        )
        plt.colorbar(img, ax=ax, label='Elevation (m)')
        ax.set_title(f'Coarse elevation  |  seed={world.seed}')
        ax.set_xlabel('j  (coarse pixel)')
        ax.set_ylabel('i  (coarse pixel)')
        ax.grid(True, color='white', alpha=0.3, linewidth=0.5)
        png_path = os.path.join(output_dir, 'coarse_elev.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved elevation preview : {png_path}")

        # ── Relief-shaded overview PNG ───────────────────────────────────────
        relief_rgb = get_relief_map(coarse_np, None, None, None,
                                    resolution=world.native_resolution * 256)
        relief_path = os.path.join(output_dir, 'coarse_relief.png')
        plt.imsave(relief_path, np.clip(relief_rgb, 0.0, 1.0))
        print(f"  Saved relief preview    : {relief_path}")

        # ── Raw elevation numpy ───────────────────────────────────────────────
        npy_path = os.path.join(output_dir, 'coarse_elev.npy')
        np.save(npy_path, coarse_np)
        print(f"  Saved elevation data    : {npy_path}")

        # ── params.json ───────────────────────────────────────────────────────
        res = world.native_resolution
        params = {
            'seed': world.seed,
            'model_path': model_path,
            'coarse_window': coarse_window,
            'coarse_offset_i': coarse_offset_i,
            'coarse_offset_j': coarse_offset_j,
            'ci0': ci0,
            'ci1': ci1,
            'cj0': cj0,
            'cj1': cj1,
            'elev_min_m': elev_min,
            'elev_max_m': elev_max,
            'native_resolution_m': res,
            'coarse_pixel_km': round(256 * res / 1000, 2),
        }
        params_path = os.path.join(output_dir, 'params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"  Saved params            : {params_path}")
        print(f"  HDF5 cache              : {hdf5_path}")
        print(f"  Elevation range         : {elev_min:.1f} m .. {elev_max:.1f} m")
        print(f"  One coarse pixel        = {256 * res / 1000:.1f} km real-world")
        print(f"  Coarse coords for detail: i=[{ci0},{ci1}), j=[{cj0},{cj1})")


# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("model_path")
@click.option("--seed", "seeds", multiple=True, type=int,
              help="Explicit seed(s) to generate (may be repeated). "
                   "Overrides --count when given.")
@click.option("--count", type=int, default=1, show_default=True,
              help="Number of random seeds to generate (ignored if --seed is provided).")
@click.option("--output-dir", default="outputs/coarse_worlds", show_default=True,
              help="Root output directory. A subdirectory per seed is created inside.")
@click.option("--coarse-window", type=int, default=50, show_default=True,
              help="Half-size of the coarse window in coarse pixels.")
@click.option("--coarse-offset-i", type=int, default=0, show_default=True,
              help="Center of the coarse window along i.")
@click.option("--coarse-offset-j", type=int, default=0, show_default=True,
              help="Center of the coarse window along j.")
@click.option("--batch-size", "batch_size_str", type=str, default="1,4", show_default=True,
              help="Batch size(s) for latent generation, e.g. '4' or '1,2,4,8'.")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose",
              show_default=True, help="Logging verbosity.")
@click.option("--compile/--no-compile", "torch_compile", default=False, show_default=True,
              help="Use torch.compile for faster inference (warm-up takes extra time).")
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32",
              show_default=True)
@click.option("--device", default=None,
              help="Compute device (cuda/cpu). Default: auto-detect.")
@click.option("--kwarg", "extra_kwargs", multiple=True,
              help="Extra key=value pipeline kwargs, e.g. --kwarg native_resolution=30")
def main(
    model_path,
    seeds,
    count,
    output_dir,
    coarse_window,
    coarse_offset_i,
    coarse_offset_j,
    batch_size_str,
    log_mode,
    torch_compile,
    dtype,
    device,
    extra_kwargs,
):
    """Generate coarse terrain map(s) and save preview images + HDF5 caches.

    MODEL_PATH is a local directory or a HuggingFace Hub model ID
    (e.g. ../terrain-diffusion-30m or xandergos/terrain-diffusion-90m).

    After reviewing the coarse_elev.png / coarse_relief.png previews, run
    generate_detail.py on the chosen seed directory to produce the
    full-resolution elevation output.
    """
    # ── Parse batch sizes ────────────────────────────────────────────────────
    if ',' in batch_size_str:
        batch_sizes = [int(x.strip()) for x in batch_size_str.split(',')]
    else:
        batch_sizes = int(batch_size_str)

    # ── Normalise dtype ──────────────────────────────────────────────────────
    if dtype == 'fp32':
        dtype = None

    # ── Device selection ──────────────────────────────────────────────────────
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("Warning: Using CPU (CUDA not available).")

    # ── Build seed list ───────────────────────────────────────────────────────
    all_seeds: list[int] = list(seeds)
    if not all_seeds:
        all_seeds = [random.randint(0, 2**31 - 1) for _ in range(max(count, 1))]

    extra = parse_kwargs(extra_kwargs)

    print(f"Generating {len(all_seeds)} world(s) into '{output_dir}/'")

    for idx, seed in enumerate(all_seeds):
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        print(f"\n{'='*64}")
        print(f"[{idx+1}/{len(all_seeds)}]  seed={seed}  →  {seed_dir}")
        print('='*64)
        generate_coarse(
            model_path=model_path,
            seed=seed,
            output_dir=seed_dir,
            coarse_window=coarse_window,
            coarse_offset_i=coarse_offset_i,
            coarse_offset_j=coarse_offset_j,
            device=device,
            batch_sizes=batch_sizes,
            log_mode=log_mode,
            torch_compile=torch_compile,
            dtype=dtype,
            **extra,
        )

    print(f"\n{'='*64}")
    print(f"Done. {len(all_seeds)} world(s) saved under '{output_dir}/'")
    print("Review the coarse_elev.png / coarse_relief.png files, then run:")
    print("  python util_scripts/generate_detail.py <model_path> <seed_dir> "
          "--ci0 CI0 --ci1 CI1 --cj0 CJ0 --cj1 CJ1")


if __name__ == '__main__':
    main()
