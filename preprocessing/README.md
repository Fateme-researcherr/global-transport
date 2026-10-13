<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download

We follow the preprocessing code used in [edm2](https://github.com/NVlabs/edm2). In this code we made a several edits: (1) we removed unncessary parts except preprocessing because this code is only used for preprocessing, (2) we use [-1, 1] range for an input to the stable diffusion VAE (similar to DiT or SiT) unlike edm2 that uses [0, 1] range, and (3) we consider preprocessing to 256x256 resolution (or 512x512 resolution).

After downloading ImageNet, please run the following scripts (please update 256x256 to 512x512 if you want to do experiments on 512x512 resolution);

```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/images --resolution=256x256 --transform=center-crop-dhariwal
```


```bash
# Convert the pixel data to VAE latents
python dataset_tools.py encode --source=[TARGET_PATH]/images \
    --dest=[TARGET_PATH]/vae-sd
```

Here,`YOUR_DOWNLOAD_PATH` is the directory that you downloaded the dataset, and `TARGET_PATH` is the directory that you will save the preprocessed images and corresponding compressed latent vectors. This directory will be used for your experiment scripts. 

The result has the layout expected by `--data-dir`:

```
TARGET_PATH/
  images/   00000/img00000000.png ...  dataset.json     (only needed for REPA and caption checks)
  vae-sd/   00000/img-mean-std-00000000.npy ...  dataset.json
```

The same data, already converted, can be downloaded with `scripts/data/download_imagenet256.sh`
(see the main README).

#### Reference statistics for FID / FD-DINOv2

`calculate_metrics.py calc` scores samples against the ImageNet-256 reference statistics `imgnet256.pkl`,
downloaded from the [DMF Huggingface repository](https://huggingface.co/kyungmnlee/DMF) into `ckpt/` at the
repository root (see the main README).
To recompute them (e.g. for another resolution or dataset):

```bash
python calculate_metrics.py ref --data=[TARGET_PATH]/images --dest=../ckpt/imgnet256.pkl
```

## Acknowledgement

This code is mainly built upon [edm2](https://github.com/NVlabs/edm2) repository.
