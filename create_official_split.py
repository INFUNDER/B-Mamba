"""
Leak-free reconstruction of the official PraNet training split.
================================================================
The official PraNet protocol trains on 900 Kvasir-SEG + 550 CVC-ClinicDB images
and tests on the *complementary* 100 Kvasir + 62 ClinicDB images found in
dataset/TestDataset/. The old `create_pranet_split.py` drew its OWN random split,
so 91/100 official Kvasir test images and 56/62 official ClinicDB test images
ended up in training (data leakage).

This script builds:
    dataset/PraNet_Official/TrainDataset/{images,masks}
  = (Kvasir-SEG  \\ TestDataset/Kvasir)  U  (CVC-ClinicDB \\ TestDataset/CVC-ClinicDB)

and verifies the result with BOTH a filename check and a perceptual pixel hash,
so no official test image can slip into training.
"""
import os
import shutil

import numpy as np
from PIL import Image

KVASIR_IMG, KVASIR_MSK = 'dataset/Kvasir-SEG/images', 'dataset/Kvasir-SEG/masks'
CLINIC_IMG, CLINIC_MSK = 'dataset/CVC-ClinicDB/images', 'dataset/CVC-ClinicDB/masks'
TEST_ROOT = 'dataset/TestDataset'
OUT_ROOT = 'dataset/PraNet_Official/TrainDataset'
EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


def stem(f):
    return os.path.splitext(f)[0]


def phash(path):
    """Simple perceptual (average) hash, robust to re-encoding png<->jpg."""
    im = Image.open(path).convert('L').resize((32, 32), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return (a > a.mean()).tobytes()


def find_mask(mask_dir, name):
    for ext in EXTS:
        p = os.path.join(mask_dir, stem(name) + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'No mask for {name} in {mask_dir}')


def build():
    img_out, msk_out = os.path.join(OUT_ROOT, 'images'), os.path.join(OUT_ROOT, 'masks')
    if os.path.exists(OUT_ROOT):
        raise SystemExit(f'{OUT_ROOT} already exists - delete it first if you want to rebuild.')
    os.makedirs(img_out)
    os.makedirs(msk_out)

    # Hashes of every official test image (all 5 test sets), used as a safety net.
    test_hashes = set()
    for ds in os.listdir(TEST_ROOT):
        d = os.path.join(TEST_ROOT, ds, 'images')
        if os.path.isdir(d):
            test_hashes |= {phash(os.path.join(d, f)) for f in list_images(d)}

    summary = {}
    for tag, img_dir, msk_dir, test_ds, expected in [
        ('kvasir', KVASIR_IMG, KVASIR_MSK, 'Kvasir', 900),
        ('clinic', CLINIC_IMG, CLINIC_MSK, 'CVC-ClinicDB', 550),
    ]:
        test_stems = {stem(f) for f in list_images(os.path.join(TEST_ROOT, test_ds, 'images'))}
        kept, dropped_name, dropped_hash = 0, 0, 0
        for f in list_images(img_dir):
            # CVC-ClinicDB frames are named 1..612; the folder also contains 5 stray
            # Kvasir images (cju*.jpg) that must not be counted as ClinicDB.
            if tag == 'clinic' and not stem(f).isdigit():
                continue
            if stem(f) in test_stems:
                dropped_name += 1
                continue
            src = os.path.join(img_dir, f)
            if phash(src) in test_hashes:  # name differs but pixels match a test image
                dropped_hash += 1
                continue
            new = f'{tag}_{f}'
            shutil.copy(src, os.path.join(img_out, new))
            m = find_mask(msk_dir, f)
            shutil.copy(m, os.path.join(msk_out, f'{tag}_{stem(f)}{os.path.splitext(m)[1]}'))
            kept += 1
        summary[tag] = (kept, dropped_name, dropped_hash, expected)

    print('=== Leak-free PraNet training split ===')
    for tag, (k, dn, dh, e) in summary.items():
        flag = 'OK' if k == e else 'CHECK'
        print(f'{tag:7s}: kept {k} (expected {e}) | removed by name {dn} | removed by pixel-hash {dh}  [{flag}]')

    # Final verification: zero overlap with any official test image.
    leaks = sum(phash(os.path.join(img_out, f)) in test_hashes for f in list_images(img_out))
    print(f'Total train images: {len(list_images(img_out))} | overlap with ANY test image: {leaks}')
    assert leaks == 0, 'Leakage detected - do not train on this split!'


if __name__ == '__main__':
    build()
