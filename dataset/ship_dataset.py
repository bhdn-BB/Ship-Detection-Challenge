import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as BaseDataset


class ShipDataset(BaseDataset):

    def __init__(self, df, images_dir, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def _rle_decode(self, rle_str, image_shape):
        s = np.fromstring(rle_str, sep=' ', dtype=int)
        starts = s[0::2] - 1
        lengths = s[1::2]
        mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
        for start, lengths in zip(starts, lengths):
            mask[start:start + lengths] = 1
        return mask.reshape(image_shape).T

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['ImageId']
        rle_list = row['EncodedPixels']
        image_path = os.path.join(self.images_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for rle in rle_list:
            if isinstance(rle, str) and rle.strip():
                mask_rle = self._rle_decode(rle, (image.shape[0], image.shape[1]))
                mask = np.maximum(mask, mask_rle)
        if self.transform:
            transform = self.transform(image=image, mask=mask)
            image, mask = transform['image'], transform['mask']
        else:
            transform = A.Compose([ToTensorV2()])
            transformed = transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        return image, mask
