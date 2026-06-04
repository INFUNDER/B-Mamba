"""
COD10K Dataset Loader — Shared
================================
Same for all 3 models. Ensures fair comparison.
Default size = 384 (cleanly divisible for all backbones).
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


class JointTransform:
    """Identical spatial transforms applied to both image and mask."""

    def __init__(self, size=384, is_train=True):
        self.size     = size
        self.is_train = is_train
        self.img_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
        )

    def __call__(self, image, mask):
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask  = mask.resize( (self.size, self.size), Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask  = mask.transpose( Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask  = mask.transpose( Image.FLIP_TOP_BOTTOM)

            if random.random() > 0.5:
                angle = random.uniform(-30, 30)
                image = image.rotate(angle, resample=Image.BILINEAR)
                mask  = mask.rotate( angle, resample=Image.NEAREST)

            if random.random() > 0.5:
                scale = random.uniform(0.7, 1.0)
                w, h  = image.size
                cw, ch= int(w*scale), int(h*scale)
                x, y  = random.randint(0, w-cw), random.randint(0, h-ch)
                image = image.crop((x,y,x+cw,y+ch)).resize((self.size,self.size), Image.BILINEAR)
                mask  = mask.crop( (x,y,x+cw,y+ch)).resize((self.size,self.size), Image.NEAREST)

            if random.random() > 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
            if random.random() > 0.7:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))

            # Random mean-fill erase
            if random.random() > 0.7:
                w, h = image.size
                ew = random.randint(int(w*0.05), int(w*0.2))
                eh = random.randint(int(h*0.05), int(h*0.2))
                ex = random.randint(0, w-ew)
                ey = random.randint(0, h-eh)
                fill = tuple(int(m*255) for m in [0.485, 0.456, 0.406])
                ImageDraw.Draw(image).rectangle([ex,ey,ex+ew,ey+eh], fill=fill)

        img_t  = transforms.ToTensor()(image)
        mask_t = (torch.from_numpy(np.array(mask)).float() > 127).float().unsqueeze(0)
        img_t  = self.img_norm(img_t)
        return img_t, mask_t


class COD10KDataset(Dataset):
    def __init__(self, root, split='Train', size=384):
        super().__init__()
        self.img_dir   = Path(root) / split / 'Image'
        self.mask_dir  = Path(root) / split / 'GT_Object'
        self.transform = JointTransform(size=size, is_train=(split=='Train'))
        self.img_paths = sorted(self.img_dir.glob('*.jpg'))
        if not self.img_paths:
            self.img_paths = sorted(self.img_dir.glob('*.png'))
        assert len(self.img_paths) > 0, f"No images found in {self.img_dir}"
        print(f"[COD10K] {split}: {len(self.img_paths)} samples  (size={size})")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        mp = self.mask_dir / (ip.stem + '.png')
        img_t, mask_t = self.transform(
            Image.open(ip).convert('RGB'),
            Image.open(mp).convert('L'),
        )
        return {'image': img_t, 'mask': mask_t, 'name': ip.stem}


def get_dataloaders(root, size=384, batch_size=8, num_workers=4):
    train_ds = COD10KDataset(root, 'Train', size)
    test_ds  = COD10KDataset(root, 'Test',  size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(test_ds,  batch_size=1, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )