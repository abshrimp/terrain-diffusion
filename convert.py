import os
import numpy as np
import glob

# ディレクトリの設定
input_dir = './data/custom_tiles'
output_dir = './data/custom_tiles_processed' # 処理後の保存先

# 保存先ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# 入力ディレクトリ内の 12_{x}_{y}.npy ファイルをすべて取得
file_pattern = os.path.join(input_dir, '12_*.npy')
file_list = glob.glob(file_pattern)

print(f"合計 {len(file_list)} 個のファイルを処理します...")

for file_path in file_list:
    # NumPy配列として読み込み
    data = np.load(file_path)
    
    # numpy.clip を使って、0未満を 0 に、3200より大きいものを 3200 に制限（クランプ）する
    # -9999.0などの欠損値もこれで一括して 0.0 に変換されます
    processed_data = np.clip(data, 0.0, 3200.0)
    
    # ファイル名を取得して新しいディレクトリに保存
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    
    np.save(output_path, processed_data)

print("すべての処理が完了しました。")
print(f"保存先: {output_dir}")