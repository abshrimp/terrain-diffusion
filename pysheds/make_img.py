# Read elevation raster
# ----------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LightSource

from pysheds.grid import Grid

print("GRID")
grid = Grid.from_raster('elevation.tif')
print("DEM")
dem = grid.read_raster('elevation.tif')


# Condition DEM
# ----------------------
# Fill pits in DEM
print("FILL PITS")
pit_filled_dem = grid.fill_pits(dem)

# Fill depressions in DEM
print("FILL DEPRESSIONS")
flooded_dem = grid.fill_depressions(pit_filled_dem)

# Resolve flats in DEM
print("INFLAT")
inflated_dem = grid.resolve_flats(flooded_dem)



# Determine D8 flow directions from DEM
# ----------------------
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

# Compute flow directions
# -------------------------------------
print("FDIR")
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)


# Calculate flow accumulation
# --------------------------
print("ACC")
acc = grid.accumulation(fdir, dirmap=dirmap)






# 流水累積量 (Flow Accumulation) の抽出
threshold = 500 
y_idx, x_idx = np.where(acc > threshold)
flow_values = acc[y_idx, x_idx]

# 座標の計算
xmin, xmax, ymin, ymax = grid.extent
lons = xmin + (xmax - xmin) * x_idx / acc.shape[1]
lats = ymax - (ymax - ymin) * y_idx / acc.shape[0]

# ポイントサイズの設定
point_sizes = np.log10(flow_values) * 3

# 標高データの欠損値処理とカラーマップの範囲設定
dem_data = np.where(inflated_dem < -100, np.nan, inflated_dem)
vmin_dem = np.nanmin(dem_data)
vmax_dem = np.nanmax(dem_data)

# 陰影（ヒルシェイド）の生成

ls = LightSource(azdeg=315, altdeg=45)
elevation_rgb = ls.shade(
    dem_data,
    cmap=plt.cm.terrain,
    blend_mode='overlay',
    vert_exag=1,
    vmin=vmin_dem,
    vmax=vmax_dem
)

# 配列のサイズ（縦・横のピクセル数）を取得
height, width = dem_data.shape

# DPIを指定し、配列サイズとぴったり合うFigureサイズを逆算
dpi = 100
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
fig.patch.set_alpha(0)

# 余白をゼロにするため、Figure全体(0~1)を占有するAxesを追加
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off') # 軸、目盛り、枠線を完全に非表示

# 背景画像（陰影付きDEM）のプロット
ax.imshow(elevation_rgb, extent=grid.extent, zorder=1, alpha=0.6, aspect='auto')

# 散布図（流水累積量）のプロット
sc = ax.scatter(
    lons, lats,
    c=flow_values,
    s=point_sizes,
    cmap='inferno',
    norm=colors.LogNorm(vmin=threshold, vmax=np.nanmax(acc)),
    zorder=2,
    alpha=0.4,
    edgecolors='none'
)

# 画像として保存 (bbox_inches='tight'はサイズが狂うので使用しない)
plt.savefig('flow_accumulation.png', dpi=dpi, transparent=True)

# メモリ解放のためにFigureを閉じる
plt.close(fig)

print(f"画像を {width}x{height} ピクセルで保存しました。")