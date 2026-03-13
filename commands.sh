# 2) .npy -> HDF5
python -m terrain_diffusion build-tiles-dataset \
  --tiles-folder data/custom_tiles \
  --output-file data/dataset_tiles.h5 \
  --resolution 30 \
  --overwrite

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
  --residual-mean 0.0 \
  --residual-std 0.7 \
  --overwrite

# 6) Decoder 学習
accelerate launch -m terrain_diffusion train \
  --config ./configs/diffusion_decoder/diffusion_decoder_64-3_tiles.cfg \
  --override training.dynamo_backend=\"no\"

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
python -m terrain_diffusion explore ../terrain-diffusion-30m-tiles