"""Caption conditions for ImageNet (continuous-condition setup of C2OT, arXiv 2503.10636).

Captions: https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched
Text encoder: open_clip hf-hub:apple/DFN5B-CLIP-ViT-H-14 (pooled, L2-normalised, 1024-d)

Needs open_clip_torch + pyarrow on top of the training environment (pip install open_clip_torch pyarrow);
see scripts/text_conditions_imagenet256.sh for the full pipeline.
"""

import io
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import torch
from tqdm import tqdm

CAPTION_REPO = 'datasets/visual-layer/imagenet-1k-vl-enriched'
CLIP_MODEL = 'hf-hub:apple/DFN5B-CLIP-ViT-H-14'
CLIP_TOKENIZER = 'ViT-H-14'


def load_clip(device):
    import open_clip
    model, preprocess = open_clip.create_model_from_pretrained(CLIP_MODEL)
    return model.to(device).eval(), preprocess, open_clip.get_tokenizer(CLIP_TOKENIZER)


def train_order(captions):
    """Row order of the training conditions = order of the edm2-style latents.

    dataset_tools.py convert indexes images by sorted source path (train/<wnid>/<image_id>.JPEG)
    and labels them by sorted wnid, so the same order is recovered from the image ids alone.
    """
    ids = sorted(captions.keys())
    wnids = sorted({i.split('_')[0] for i in ids})
    wnid_to_label = {w: k for k, w in enumerate(wnids)}
    labels = np.array([wnid_to_label[i.split('_')[0]] for i in ids], dtype=np.int64)
    return ids, labels


def latent_labels(data_dir):
    with open(os.path.join(data_dir, 'vae-sd', 'dataset.json'), 'rb') as f:
        labels = json.load(f)['labels']
    return np.array([label for _fname, label in sorted(labels)], dtype=np.int64)


def open_images(data_dir):
    """Training images from images/ if it is unpacked, otherwise straight from images.zip.
    Returns the sorted image file names, their labels and a function reading a file's bytes."""
    images_dir = os.path.join(data_dir, 'images')
    if os.path.isfile(os.path.join(images_dir, 'dataset.json')):
        def read(fname):
            with open(os.path.join(images_dir, fname), 'rb') as f:
                return f.read()
    else:
        read = zipfile.ZipFile(os.path.join(data_dir, 'images.zip')).read
    entries = sorted(json.loads(read('dataset.json'))['labels'])
    fnames = [fname for fname, _label in entries]
    labels = np.array([label for _fname, label in entries], dtype=np.int64)
    return fnames, labels, read


#----------------------------------------------------------------------------

@click.group()
def cmdline():
    pass


@cmdline.command()
@click.option('--split', help='Caption split', type=click.Choice(['train', 'validation']), required=True)
@click.option('--dest',  help='Output json {image_id: caption}', metavar='JSON', type=str, required=True)
@click.option('--workers', help='Parallel shard downloads', metavar='INT', type=int, default=8, show_default=True)
def captions(split, dest, workers):
    """Download the captions (only the id/caption columns of the parquet shards are read)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shards = sorted(fs.glob(f'{CAPTION_REPO}/data/{split}-*.parquet'))

    def read_shard(shard):
        with fs.open(shard, 'rb') as f:
            table = pq.read_table(f, columns=['image_id', 'caption_enriched'])
        return table['image_id'].to_pylist(), table['caption_enriched'].to_pylist()

    out = {}
    with ThreadPoolExecutor(workers) as pool:
        for image_ids, caps in tqdm(pool.map(read_shard, shards), total=len(shards)):
            for image_id, caption in zip(image_ids, caps):
                out[image_id] = (caption or '').strip()
    print(f'{len(out)} captions, {sum(c == "" for c in out.values())} empty')
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, 'w') as f:
        json.dump(out, f)


@cmdline.command()
@click.option('--captions', 'captions_path', help='Input json {image_id: caption}', metavar='JSON', type=str, required=True)
@click.option('--dest',     help='Output (N, 1024) float16 .npy; <dest>.ids.json holds the row ids', metavar='NPY', type=str, required=True)
@click.option('--data-dir', help='Training data dir (with vae-sd/): order the rows like the latents and check the labels', metavar='PATH', type=str)
@click.option('--batch',    help='Batch size', metavar='INT', type=int, default=1024, show_default=True)
def encode(captions_path, dest, data_dir, batch):
    """Encode captions into text embeddings, one row per image."""
    with open(captions_path) as f:
        caps = json.load(f)

    if data_dir is not None:
        ids, labels = train_order(caps)
        ref = latent_labels(data_dir)
        if len(ref) != len(labels):
            raise click.ClickException(f'{len(labels)} captions but {len(ref)} latents.')
        if (ref != labels).any():
            raise click.ClickException(f'Label mismatch at {(ref != labels).sum()} rows: the latents are not in sorted-path order.')
        print(f'Labels of all {len(ids)} rows match vae-sd/dataset.json.')
    else:
        ids = sorted(caps.keys())

    device = torch.device('cuda')
    model, _preprocess, tokenizer = load_clip(device)
    embs = np.zeros((len(ids), model.text_projection.shape[-1]), dtype=np.float16)
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in tqdm(range(0, len(ids), batch)):
            tokens = tokenizer([caps[k] for k in ids[i:i + batch]]).to(device)
            embs[i:i + batch] = model.encode_text(tokens, normalize=True).float().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    np.save(dest, embs)
    with open(dest + '.ids.json', 'w') as f:
        json.dump(ids, f)
    print(f'Saved {embs.shape} to {dest}')


@cmdline.command()
@click.option('--conds',    help='Training conditions from `encode`', metavar='NPY', type=str, required=True)
@click.option('--data-dir', help='Training data dir (with vae-sd/ and images/ or images.zip)', metavar='PATH', type=str, required=True)
@click.option('--num',      help='Number of images to check', metavar='INT', type=int, default=2000, show_default=True)
@click.option('--seed',     help='Random seed', metavar='INT', type=int, default=0, show_default=True)
def verify(conds, data_dir, num, seed):
    """Check the row alignment inside classes: an image should be closer to its own caption
    embedding than to the one of its neighbouring row (same class, since rows are class-sorted)."""
    import PIL.Image

    fnames, labels, read = open_images(data_dir)
    embs = np.load(conds, mmap_mode='r')
    assert len(embs) == len(fnames), f'{len(embs)} conditions but {len(fnames)} images.'
    assert (labels == latent_labels(data_dir)).all(), 'images and latents are not in the same order.'

    device = torch.device('cuda')
    model, preprocess, _tokenizer = load_clip(device)
    idx = np.sort(np.random.RandomState(seed).choice(len(fnames) - 1, num, replace=False))
    own, other = [], []
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in tqdm(range(0, num, 256)):
            b = idx[i:i + 256]
            imgs = torch.stack([preprocess(PIL.Image.open(io.BytesIO(read(fnames[k]))).convert('RGB')) for k in b])
            z = model.encode_image(imgs.to(device), normalize=True).float().cpu()
            own.append((z * torch.from_numpy(embs[b].astype(np.float32))).sum(-1))
            other.append((z * torch.from_numpy(embs[b + 1].astype(np.float32))).sum(-1))
    own, other = torch.cat(own), torch.cat(other)
    print(f'image-text cosine: own caption {own.mean():.4f}, next row {other.mean():.4f}, own > next in {(own > other).float().mean():.1%} of {num}')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()
