import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class PolypDataset(Dataset):
    def __init__(self, image_root, gt_root, trainsize=352, is_train=True, strong_aug=False):
        self.trainsize = trainsize
        self.is_train = is_train
        self.strong_aug = strong_aug and is_train
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        # Filter out non-image files like .DS_Store
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        assert len(self.images) == len(self.gts), f"Number of images ({len(self.images)}) and masks ({len(self.gts)}) do not match!"
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        image = image.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)
        
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt = TF.hflip(gt)
            if random.random() > 0.5:
                image = TF.vflip(image)
                gt = TF.vflip(gt)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                gt = TF.rotate(gt, angle)
            if self.strong_aug:
                image, gt = self._strong_aug(image, gt)
                
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        gt = TF.to_tensor(gt)
        
        return image, gt

    def _strong_aug(self, image, gt):
        # Appearance: simulates different scopes / lighting (helps unseen domains)
        if random.random() < 0.8:
            image = self.color_jitter(image)
        if random.random() < 0.2:
            image = TF.gaussian_blur(image, kernel_size=5, sigma=random.uniform(0.1, 1.5))
        # Geometry: random scale + translation + small rotation (zoom in/out on polyps)
        if random.random() < 0.7:
            s = self.trainsize
            angle = random.uniform(-25, 25)
            translate = (int(random.uniform(-0.1, 0.1) * s), int(random.uniform(-0.1, 0.1) * s))
            scale = random.uniform(0.75, 1.35)
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0.0,
                              interpolation=TF.InterpolationMode.BILINEAR)
            gt = TF.affine(gt, angle=angle, translate=translate, scale=scale, shear=0.0,
                           interpolation=TF.InterpolationMode.NEAREST)
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.images)

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, is_train=True):
    dataset = PolypDataset(image_root, gt_root, trainsize, is_train)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batchsize,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return data_loader
