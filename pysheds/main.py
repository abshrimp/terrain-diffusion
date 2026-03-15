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



# 陰影図描画
fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_alpha(0)

vmin = np.nanmin(inflated_dem)
vmax = np.nanmax(inflated_dem)

ls = LightSource(azdeg=315, altdeg=45)

shaded_dem = ls.shade(
    inflated_dem, 
    cmap=plt.cm.terrain, 
    blend_mode='overlay',
    vmin=vmin, 
    vmax=vmax
)

ax.imshow(shaded_dem, extent=grid.extent, zorder=1)

sm = plt.cm.ScalarMappable(cmap='terrain', norm=plt.Normalize(vmin=vmin, vmax=vmax))
plt.colorbar(sm, ax=ax, label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

# plt.imsave('shaded_dem_output.png', shaded_dem)



# Determine D8 flow directions from DEM
# ----------------------
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

# Compute flow directions
# -------------------------------------
print("FDIR")
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid', size=14)
plt.grid(zorder=-1)
plt.tight_layout()
plt.show()




# Calculate flow accumulation
# --------------------------
print("ACC")
acc = grid.accumulation(fdir, dirmap=dirmap)

# fig, ax = plt.subplots(figsize=(16,12))
# fig.patch.set_alpha(0)
# plt.grid('on', zorder=0)
# im = ax.imshow(acc, extent=grid.extent, zorder=2,
#                cmap='cubehelix',
#                norm=colors.LogNorm(1, acc.max()),
#                interpolation='bilinear')
# plt.colorbar(im, ax=ax, label='Upstream Cells')
# plt.title('Flow Accumulation', size=14)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.tight_layout()
# plt.show()



# threshold = 1000 
# y_idx, x_idx = np.where(acc > threshold)
# flow_values = acc[y_idx, x_idx]

# xmin, xmax, ymin, ymax = grid.extent
# lons = xmin + (xmax - xmin) * x_idx / acc.shape[1]
# lats = ymax - (ymax - ymin) * y_idx / acc.shape[0] 

# fig, ax = plt.subplots(figsize=(16,12))
# fig.patch.set_alpha(0)
# plt.grid('on', zorder=0)
# ax.set_aspect('equal')

# point_sizes = np.log10(flow_values) * 3 

# sc = ax.scatter(
#     lons, lats,
#     c=flow_values,
#     s=point_sizes,
#     cmap='Blues',
#     norm=colors.LogNorm(vmin=threshold, vmax=np.nanmax(acc)),
#     zorder=2,
#     alpha=0.8,
#     edgecolors='none'
# )

# plt.colorbar(sc, ax=ax, label='Upstream Cells (Flow Accumulation)')
# plt.title('Flow Accumulation (Scatter Map)', size=14)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.tight_layout()
# plt.show()


threshold = 1000 
y_idx, x_idx = np.where(acc > threshold)
flow_values = acc[y_idx, x_idx]

xmin, xmax, ymin, ymax = grid.extent
lons = xmin + (xmax - xmin) * x_idx / acc.shape[1]
lats = ymax - (ymax - ymin) * y_idx / acc.shape[0]

point_sizes = np.log10(flow_values) * 1

dem_data = np.where(inflated_dem < -100, np.nan, inflated_dem)
vmin_dem = np.nanmin(dem_data)
vmax_dem = np.nanmax(dem_data)

ls = LightSource(azdeg=315, altdeg=45)

elevation_rgb = ls.shade(
    dem_data,
    cmap=plt.cm.terrain,
    blend_mode='overlay',
    vert_exag=1,
    vmin=vmin_dem,
    vmax=vmax_dem
)

fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0, color='gray', linestyle='--', alpha=0.5)
ax.set_aspect('equal')
ax.imshow(elevation_rgb, extent=grid.extent, zorder=1, alpha=0.6)

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

plt.colorbar(sc, ax=ax, label='Upstream Cells (Flow Accumulation)')
plt.title('Flow Accumulation over Shaded Elevation', size=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.show()



# Calculate branches
# --------------------------
# branches = grid.extract_river_network(fdir, acc > 1000, dirmap=dirmap)

# sns.set_palette('husl')
# fig, ax = plt.subplots(figsize=(16,12))

# plt.xlim(grid.bbox[0], grid.bbox[2])
# plt.ylim(grid.bbox[1], grid.bbox[3])
# ax.set_aspect('equal')

# for branch in branches['features']:
#     line = np.asarray(branch['geometry']['coordinates'])
#     plt.plot(line[:, 0], line[:, 1])
    
# _ = plt.title('D8 channels', size=14)




# 標高 + acc 画像保存

threshold = 1000 
y_idx, x_idx = np.where(acc > threshold)
flow_values = acc[y_idx, x_idx]

xmin, xmax, ymin, ymax = grid.extent
lons = xmin + (xmax - xmin) * x_idx / acc.shape[1]
lats = ymax - (ymax - ymin) * y_idx / acc.shape[0]

point_sizes = np.log10(flow_values) * 3

dem_data = np.where(inflated_dem < -100, np.nan, inflated_dem)
vmin_dem = np.nanmin(dem_data)
vmax_dem = np.nanmax(dem_data)

ls = LightSource(azdeg=315, altdeg=45)
elevation_rgb = ls.shade(
    dem_data,
    cmap=plt.cm.terrain,
    blend_mode='overlay',
    vert_exag=1,
    vmin=vmin_dem,
    vmax=vmax_dem
)

height, width = dem_data.shape

dpi = 100
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
fig.patch.set_alpha(0)

ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

ax.imshow(elevation_rgb, extent=grid.extent, zorder=1, alpha=0.6, aspect='auto')

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

plt.savefig('flow_accumulation.png', dpi=dpi, transparent=True)
plt.close(fig)
