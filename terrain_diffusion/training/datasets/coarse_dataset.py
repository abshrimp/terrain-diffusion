import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CoarseDataset(Dataset):
    """
    カスタムの .npy タイルデータ（標高1チャンネル）を読み込むデータセット。
    拡散モデルの学習に必要なノイズ付加（cond_img, cond_inputs）などの処理は元コードに準拠。
    """
    def __init__(self, 
                 data_dir="data/custom_tiles",
                 crop_size=16, # 学習時のパッチサイズ。config等で指定される場合はそちらが優先されます
                 sigma_data=0.5,
                 **kwargs): # configから余分な引数が渡されてもエラーにならないようにkwargsを追加
        
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.sigma_data = sigma_data
        
        # タイルデータのパスを取得
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "12_*.npy")))
        if len(self.file_paths) == 0:
            print(f"Warning: No .npy files found in {self.data_dir}")
            
        self.max_elevation = 3200.0

    def __len__(self):
        """
        元のコード同様、エポックごとのイテレーション数を固定値にするか、
        データ数にするか選べます。ここでは実際のファイル数とします。
        """
        return len(self.file_paths)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, idx):
        # 1. データの読み込みと前処理
        # 元コードはクロップを行っていましたが、npyが既に256x256タイルであるため、
        # ここではタイルを読み込み、必要に応じてcrop_sizeで切り出します。
        data = np.load(self.file_paths[idx])
        
        # 欠損値(-9999.0)や負の値を0にし、最大値を3200に制限
        data = np.clip(data, 0.0, self.max_elevation)
        
        # ランダムクロップの処理（学習を安定させるため、元の仕様に合わせてランダムに切り出す）
        h, w = data.shape
        if h > self.crop_size and w > self.crop_size:
            i = random.randint(0, h - self.crop_size)
            j = random.randint(0, w - self.crop_size)
            crop = data[i:i+self.crop_size, j:j+self.crop_size]
        else:
            # 万が一データがcrop_sizeより小さい場合のフォールバック（通常は起こりません）
            crop = data[:self.crop_size, :self.crop_size]
        
        # 拡散モデル向けに [-1, 1] の範囲に正規化
        crop_normalized = (crop / (self.max_elevation / 2.0)) - 1.0
        
        # PyTorchのTensorに変換し、チャンネル次元を追加: (1, crop_size, crop_size)
        tensor_data = torch.from_numpy(crop_normalized).float().unsqueeze(0)
        
        # 2. データ拡張（Data Augmentation） - 元コード準拠
        if random.random() > 0.5:
            tensor_data = torch.flip(tensor_data, dims=[-2]) # 上下反転
        k = random.randint(0, 3)
        if k > 0:
            tensor_data = torch.rot90(tensor_data, k=k, dims=[-2, -1]) # 90度回転
            
        tensor_data = tensor_data * self.sigma_data
        
        # 3. 拡散モデル用の条件付けとノイズ処理 - 元コード準拠
        # 元コードは6チャンネル（標高、気候等）を想定し、1チャンネル(標高)以外をcond_imgとしていましたが、
        # 今回は1チャンネルしかありません。そのため、cond_imgを生成するための元のロジックを1チャンネル用に調整します。
        
        # 元コード: t = torch.atan(torch.exp(10 * torch.rand(5) - 5)).view(-1, 1, 1)
        # 今回はチャンネルが1つしかないので、時間ステップ t も1つ（あるいはモデルの入力仕様に合わせてゼロベクトル等）にします。
        # terrain-diffusionのアーキテクチャに依存しますが、一般的には条件付け画像がない場合は、cond_imgは不要か、ゼロテンソルを渡します。
        # ここでは安全のため、元コードのノイズ付加ロジックを「自己条件付け」や「低解像度からのアップサンプリング条件付け」のダミーとして簡略化します。
        
        # [重要] 1チャンネル入力のみの場合、条件付け（cond_img, cond_inputs）をどう扱うかはモデル構造に強く依存します。
        # 以下の実装は、モデルが「cond_img」と「cond_inputs」を受け取ることを前提としたダミー処理です。
        # もし `diffusion_coarse.cfg` で in_channels=1 に変更しているなら、cond_imgのチャンネル数もそれに合わせる必要があります。
        
        # 時間ステップのサンプリング (1次元)
        t = torch.atan(torch.exp(10 * torch.rand(1) - 5)).view(-1, 1, 1)
        
        # --- [修正ポイント] ---
        # cond_img が意図せず複数チャンネルにならないよう、元の tensor_data と
        # まったく同じ形状（1チャンネル）であることを保証します。
        cond_img = tensor_data.clone() / self.sigma_data
        cond_img = cond_img * torch.cos(t) + torch.randn_like(cond_img) * torch.sin(t)
        
        cond_inputs = [torch.log(torch.tan(s) / 8) for s in t.flatten()]

        # print("DEBUG Shapes:", tensor_data.shape, cond_img.shape) # 不安ならデバッグ用にprintを入れてもOK

        return {
            'image': tensor_data,      # (1, 16, 16) であるべき
            'cond_img': cond_img,      # (1, 16, 16) であるべき
            'cond_inputs': cond_inputs
        }

if __name__ == "__main__":
    # テスト用の実行コード
    dataset = CoarseDataset(data_dir='data/custom_tiles', crop_size=32)
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Image shape:", sample['image'].shape)
        print("Cond_img shape:", sample['cond_img'].shape)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].matshow(sample['image'][0].numpy(), vmin=-1, vmax=1)
        axes[0].set_title('Image')
        axes[1].matshow(sample['cond_img'][0].numpy())
        axes[1].set_title('Cond_img')
        plt.show()
    else:
        print("No data available to plot.")