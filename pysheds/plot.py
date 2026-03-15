import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio
from rasterio.transform import from_origin
import math

FILE = "../outputs/coarse_worlds/seed_12345/detail_ci-29_-15_cj-2_16/elevation.npy"
OUT_IMG = "elevation.png"
OUT_TIF = "elevation.tif"

# 左上座標
UL_LAT = 32.537452
UL_LON = 128.915168

# 緯度経度で30mに対応するピクセルサイズ
pixel_size_lat = 30 / 111320  # degree
pixel_size_lon = 30 / (111320 * math.cos(math.radians(UL_LAT)))

# 読み込み
elevation = np.load(FILE)

# NPY概要
print("=== NPY Info ===")
print("shape:", elevation.shape)
print("dtype:", elevation.dtype)
print("min:", np.nanmin(elevation))
print("max:", np.nanmax(elevation))
print("mean:", np.nanmean(elevation))
print("nan count:", np.isnan(elevation).sum())

# 欠損値処理
elevation[elevation == -9999] = np.nan

# ===== PNG描画用 =====
cmap = plt.get_cmap("terrain").copy()
cmap.set_under("white")
norm = colors.Normalize(vmin=0, vmax=np.nanmax(elevation))

plt.figure(figsize=(6,6))
img = plt.imshow(elevation, cmap=cmap, norm=norm)
cbar = plt.colorbar(img)
cbar.set_label("Elevation (m)")
plt.axis("off")
plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
plt.close()
print("saved:", OUT_IMG)

# ===== GeoTIFF用 =====
elevation_tif = elevation.copy()
# 0未満は欠損値に
elevation_tif[elevation_tif < 0] = np.nan

h, w = elevation_tif.shape
transform = from_origin(
    UL_LON,
    UL_LAT,
    pixel_size_lon,
    pixel_size_lat
)

with rasterio.open(
    OUT_TIF,
    "w",
    driver="GTiff",
    height=h,
    width=w,
    count=1,
    dtype=elevation_tif.dtype,
    crs="EPSG:4326",
    transform=transform,
    nodata=np.nan
) as dst:
    dst.write(elevation_tif, 1)

print("saved:", OUT_TIF)