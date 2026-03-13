#!/usr/bin/env python3
"""
japan_dataset.h5 の各チャンクに擬似 climate データセットを追加する。

Base モデルの H5LatentsDataset は climate[0,3,11,14] (BIO1,BIO4,BIO12,BIO15) を
読み取る。ここでは日本の平均的な気候値を定数として埋める。
cond_input_dropout=1.0 にすれば訓練時に無視されるが、
データ構造として必要なため追加する。

使い方:
  cd terrain-diffusion
  python data/add_climate_to_japan_dataset.py --h5-file data/japan_dataset.h5
"""

import click
import h5py
import numpy as np
from tqdm import tqdm

# 日本の代表的な気候値 (WorldClim BIO 変数, 各インデックスに対応)
# インデックスは 0 始まり (BIO1=idx0, BIO2=idx1, ...)
JAPAN_CLIMATE = {
    0:  12.0,   # BIO1:  年平均気温 (°C)
    1:   9.0,   # BIO2:  平均日較差
    2:  30.0,   # BIO3:  等温性
    3: 700.0,   # BIO4:  気温季節性
    4:  26.0,   # BIO5:  最暖月最高気温
    5:  -3.0,   # BIO6:  最寒月最低気温
    6:  29.0,   # BIO7:  気温年較差
    7:  22.0,   # BIO8:  最湿四半期平均気温
    8:   3.0,   # BIO9:  最乾四半期平均気温
    9:  22.0,   # BIO10: 最暖四半期平均気温
    10:  2.0,   # BIO11: 最寒四半期平均気温
    11: 1700.0, # BIO12: 年降水量 (mm)
    12: 220.0,  # BIO13: 最湿月降水量
    13:  55.0,  # BIO14: 最乾月降水量
    14:  70.0,  # BIO15: 降水季節性 (変動係数)
}
N_CLIMATE_CHANNELS = 15  # インデックス 14 まで必要


@click.command()
@click.option("--h5-file", default="data/japan_dataset.h5", show_default=True,
              help="対象 HDF5 ファイル")
@click.option("--overwrite", is_flag=True, default=False,
              help="既存の climate データを上書きする")
def main(h5_file: str, overwrite: bool):
    """japan_dataset.h5 に擬似 climate データを追加する。"""

    # 定数 climate 配列を作成 (ブロードキャスト用に 1x1 サイズ; 各チャンクで拡張)
    climate_values = np.array(
        [JAPAN_CLIMATE.get(i, 0.0) for i in range(N_CLIMATE_CHANNELS)],
        dtype=np.float32
    )

    with h5py.File(h5_file, "a") as f:
        for res_key in f.keys():
            res_group = f[res_key]
            chunk_ids = list(res_group.keys())
            for chunk_id in tqdm(chunk_ids, desc=f"res={res_key}"):
                chunk_group = res_group[chunk_id]
                for subchunk_id in chunk_group.keys():
                    sg = chunk_group[subchunk_id]

                    if "climate" in sg:
                        if not overwrite:
                            continue
                        del sg["climate"]

                    # lowres_exact と同じ空間サイズを使用
                    if "lowres_exact" not in sg:
                        continue
                    H, W = sg["lowres_exact"].shape

                    # [N_CLIMATE_CHANNELS, H, W] の定数テンソルを作成
                    climate = np.broadcast_to(
                        climate_values[:, None, None],
                        (N_CLIMATE_CHANNELS, H, W)
                    ).copy().astype(np.float32)

                    ds = sg.create_dataset(
                        "climate", data=climate,
                        compression="lzf",
                        chunks=(N_CLIMATE_CHANNELS, min(16, H), min(16, W))
                    )
                    # 属性を residual から引き継ぐ
                    if "residual" in sg:
                        ds.attrs.update(sg["residual"].attrs)

    print(f"\n完了: {h5_file} に climate データを追加しました。")
    print(f"  チャンネル数 : {N_CLIMATE_CHANNELS}")
    print(f"  使用した値   : BIO1={JAPAN_CLIMATE[0]}, BIO4={JAPAN_CLIMATE[3]}, "
          f"BIO12={JAPAN_CLIMATE[11]}, BIO15={JAPAN_CLIMATE[14]}")


if __name__ == "__main__":
    main()
