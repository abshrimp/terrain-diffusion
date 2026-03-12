import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class CoarseDataset(Dataset):
    """
    Japanese DEM (.npy) 用のカスタム CoarseDataset。
    設定ファイルからの不要な引数は無視し、ローカルのタイルデータを読み込んで
    元の学習パイプラインが要求する11チャンネルの入出力形式を再現します。
    """
    def __init__(self, **kwargs):
        # .cfg から渡される h5_file や etopo_file などの引数は受け取りますが、無視します。
        self.data_dir = './data/custom_tiles_processed'
        self.file_list = glob.glob(os.path.join(self.data_dir, '*.npy'))
        self.crop_size = kwargs.get('crop_size', 16) # デフォルトは16
        self.sigma_data = kwargs.get('sigma_data', 0.5)
        self.max_elev = 3200.0
        
        if len(self.file_list) == 0:
            print(f"Warning: {self.data_dir} に .npy ファイルが見つかりません。")

    def __len__(self):
        # 元の実装に合わせて、1エポックあたりのステップ数を維持するために十分な数を返します
        return 10000 

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        # 1. ランダムにタイルを1枚選んで読み込む
        file_path = random.choice(self.file_list)
        data_np = np.load(file_path) # (256, 256)
        
        # 2. 16x16 (crop_size) のランダムクロップを切り出す
        h, w = data_np.shape
        max_i = h - self.crop_size
        max_j = w - self.crop_size
        
        if max_i > 0 and max_j > 0:
            i = random.randint(0, max_i)
            j = random.randint(0, max_j)
            crop = data_np[i:i+self.crop_size, j:j+self.crop_size]
        else:
            crop = data_np[:self.crop_size, :self.crop_size]
            
        # 3. 標高を正規化 (0~3200m をおおよそ -1.0 ~ 1.0 にスケーリング)
        elev_norm = (crop / (self.max_elev / 2.0)) - 1.0
        
        # 4. モデルが要求する6チャンネルのTensorを作成
        # Ch 0: 標高 (Elevation mean)
        # Ch 1: 標高の粗さ (Elevation p5 - 便宜上 0.0 で埋める)
        # Ch 2~5: 気候データ (ダミーとして全て 0.0)
        data = torch.zeros((6, self.crop_size, self.crop_size), dtype=torch.float32)
        data[0] = torch.from_numpy(elev_norm)
        # Ch 1~5 は初期化時の zeros のまま
        
        # 5. データオーグメンテーション (反転・回転)
        if random.random() > 0.5:
            data = torch.flip(data, dims=[-2])
        k = random.randint(0, 3)
        if k > 0:
            data = torch.rot90(data, k=k, dims=[-2, -1])
            
        # スケーリング (元のコードを踏襲)
        data = data * self.sigma_data
        
        # 6. 条件付き入力 (cond_img) の生成処理 (元のアルゴリズムを完全再現)
        # Ch 1 を除外した 5チャンネル分を取り出す
        t = torch.atan(torch.exp(10 * torch.rand(5) - 5)).view(-1, 1, 1)
        cond_img = data[[0, 2, 3, 4, 5]] / self.sigma_data
        cond_img = cond_img * torch.cos(t) + torch.randn_like(cond_img) * torch.sin(t)
        
        return {
            'image': data, 
            'cond_img': cond_img, 
            'cond_inputs': [torch.log(torch.tan(s) / 8) for s in t.flatten()]
        }