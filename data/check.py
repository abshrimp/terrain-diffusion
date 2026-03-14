import os
import re
import json

# 設定: .npyファイルが格納されているディレクトリ
data_dir = './custom_tiles/'
output_file = 'tile_viewer.html'

# ファイル名のパターン (12_x_y.npy)
pattern = re.compile(r'12_(\d+)_(\d+)\.npy')

tile_coords = []

# ディレクトリ内のファイルをスキャン
if os.path.exists(data_dir):
    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if match:
            x, y = map(int, match.groups())
            tile_coords.append({'x': x, 'y': y})

print(f"検出されたタイル数: {len(tile_coords)}")

# HTMLテンプレート
html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tile Range Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        #map { height: 95vh; width: 100%; }
        body { margin: 0; padding: 0; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const tileData = %TILE_DATA%;
        const zoom = 12;

        const map = L.map('map').setView([0, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        }).addTo(map);

        function tileToBounds(x, y, z) {
            const n = Math.pow(2, z);
            const unit = 360 / n;
            const lon = x * unit - 180;
            const lonNext = (x + 1) * unit - 180;
            
            function yToLat(yVal) {
                const n2 = Math.PI - 2 * Math.PI * yVal / n;
                return 180 / Math.PI * Math.atan(0.5 * (Math.exp(n2) - Math.exp(-n2)));
            }
            return [[yToLat(y + 1), lon], [yToLat(y), lonNext]];
        }

        // 描画の最適化: Canvasを使用して大量のタイルを高速描画
        const canvasRenderer = L.canvas({ padding: 0.5 });
        
        const rects = tileData.map(tile => {
            return L.rectangle(tileToBounds(tile.x, tile.y, zoom), {
                color: "#ff3300", weight: 1, fillOpacity: 0.2, renderer: canvasRenderer
            });
        });

        const group = L.featureGroup(rects).addTo(map);
        if (tileData.length > 0) {
            map.fitBounds(group.getBounds());
        }
    </script>
</body>
</html>
"""

# HTMLの書き出し
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_template.replace('%TILE_DATA%', json.dumps(tile_coords)))

print(f"完了: {output_file} を作成しました。")