# FADER:Frequency ADaptive implicit nEuRal representations

To better verify the advantages of FADER for representing high-frequency components, we follow the experimental setting from [WIRE](https://arxiv.org/pdf/2301.05187) (Sec. 4.3), where only 25 images are used for training, and each image is downsampled to a resolution of $200\times200$.

Thanks to [torch-ngp](https://github.com/ashawkey/torch-ngp) and [finer] (https://github.com/liuzhen0212/FINER)

Finally, please add the **[network_siren_1.py](network_siren_1.py)** file to the [torch-ngp/nerf](https://github.com/ashawkey/torch-ngp/tree/main/nerf) directory, and use **[provider.py](provider.py)** and **[main_nerf.py](main_nerf.py)** files, respectively. 

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py ../data/nerf_synthetic/drums \
    --nn finer --lr 2e-4 --iter 37500 --downscale 4 \
    --trainskip 4 \
    --num_layers 4 --hidden_dim 182 --geo_feat_dim 182 --num_layers_color 4 --hidden_dim_color 182 \
    --workspace logs/drums_finer \
    -O --bound 1 --scale 0.8 --dt_gamma 0
```
