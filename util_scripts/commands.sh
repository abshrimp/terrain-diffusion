# 2) .npy -> HDF5
python -m terrain_diffusion build-tiles-dataset \
  --tiles-folder data/custom_tiles \
  --output-file data/dataset_tiles.h5 \
  --resolution 30 \
  --overwrite

# 2.5) Residual stats を config に反映

# 3) AutoEncoder 学習
python -m terrain_diffusion train \
  --config ./configs/autoencoder/autoencoder_x8_tiles.cfg

# 4) AutoEncoder を推論用に保存
python -m terrain_diffusion.training.save_model \
  -c checkpoints/autoencoder_x8_tiles/latest_checkpoint \
  -s 0.05

mkdir -p checkpoints/models/autoencoder_x8_tiles
mv checkpoints/autoencoder_x8_tiles/latest_checkpoint/saved_model/* \
   checkpoints/models/autoencoder_x8_tiles/

# 5) latent 生成
python -m terrain_diffusion build-encoded-dataset \
  --dataset data/dataset_tiles.h5 \
  --resolution 30 \
  --encoder ./checkpoints/models/autoencoder_x8_tiles \
  --use-fp16 \
  --compile-model \
  --residual-mean -0.001008 \
  --residual-std 1.652817 \
  --overwrite

sudo mount -o remount,size=8G /dev/shm

# 6) Decoder 学習
accelerate launch -m terrain_diffusion train \
  --config ./configs/diffusion_decoder/diffusion_decoder_64-3_tiles.cfg \
  --override training.dynamo_backend=\"no\" \
  --override training.mixed_precision=\"bf16\" \
  --override training.batch_size=1 \
  --override training.epochs=200 \
  --override lr_sched.@lr_sched=\"cosine_window\" \
  --override lr_sched.decay_start_epoch=100 \
  --override lr_sched.decay_end_epoch=200 \
  --override lr_sched.warmup_epochs=10 \
  --override lr_sched.min_lr_ratio=0.0 \
  --override evaluation.validate_epochs=100

# 7) Decoder を推論用に保存
python -m terrain_diffusion.training.save_model \
  -c checkpoints/diffusion_decoder-64x3_tiles/latest_checkpoint \
  -s 0.05

mkdir -p checkpoints/models/diffusion_decoder-64x3_tiles
mv checkpoints/diffusion_decoder-64x3_tiles/latest_checkpoint/saved_model/* \
   checkpoints/models/diffusion_decoder-64x3_tiles/

# 8) Base 学習
accelerate launch -m terrain_diffusion train \
  --config ./configs/diffusion_base/diffusion_192-3_tiles.cfg

# 9) Base を推論用に保存
python -m terrain_diffusion.training.save_model \
  -c checkpoints/diffusion_base-192x3_tiles/latest_checkpoint \
  -s 0.05

mkdir -p checkpoints/models/diffusion_base-192x3_tiles
mv checkpoints/diffusion_base-192x3_tiles/latest_checkpoint/saved_model/* \
   checkpoints/models/diffusion_base-192x3_tiles/



# 10) 既存30mモデルを複製（coarseはそのまま使う）
cp -R terrain-diffusion-30m terrain-diffusion-30m-tiles

# 11) base/decoder を学習済みで差し替え
rm -rf terrain-diffusion-30m-tiles/base_model/*
rm -rf terrain-diffusion-30m-tiles/decoder_model/*

cp -R terrain-diffusion/checkpoints/models/diffusion_base-192x3_tiles/* \
      terrain-diffusion-30m-tiles/base_model/
cp -R terrain-diffusion/checkpoints/models/diffusion_decoder-64x3_tiles/* \
      terrain-diffusion-30m-tiles/decoder_model/

# 12) explore 実行
python -m terrain_diffusion explore ../terrain-diffusion-tiles --seed 0




# ランダムseedを3個生成
python util_scripts/generate_coarse.py ../terrain-diffusion-30m \
    --count 3 \
    --output-dir outputs/coarse_worlds

# または明示的なseedを指定
python util_scripts/generate_coarse.py ../terrain-diffusion-30m \
    --seed 12345 --seed 67890


# coarse_elev.png のX軸(j)・Y軸(i)を見て範囲を指定
python util_scripts/generate_detail.py \
    ../terrain-diffusion-30m \
    outputs/coarse_worlds/seed_12345 \
    --ci0 10 --ci1 12 --cj0 10 --cj1 12
    # --hydro-enforce

# 指定した範囲のフォルダに以下が出力される:
#   elevation.npy   — float32 (H,W) 標高データ [m]
#   relief.png      — 陰影起伏図
#   elevation.png   — terrain カラーマップ
#   result_params.json