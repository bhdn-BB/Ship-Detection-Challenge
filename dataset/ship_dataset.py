import os
from typing import Optional

import numpy as np
from PIL import Image
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset as BaseDataset
import pandas as pd


class ShipDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: Optional[Compose] = None
    ) -> None:
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def __rle_decode(
            self,
            rle_str: str,
            image_shape: tuple[int, int]
    ) -> np.ndarray:
        s = np.fromstring(rle_str, sep=' ', dtype=int)
        starts = s[0::2] - 1
        lengths = s[1::2]
        mask = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1
        return mask.reshape(image_shape, order='F')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]
        image_id: str = row['ImageId']
        rle_list = row['EncodedPixels']
        image_path = os.path.join(self.images_dir, image_id)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size  #(width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        for rle in rle_list:
            mask_rle = self.__rle_decode(rle, (height, width))
            mask = np.maximum(mask, mask_rle)
        if self.transform:
            transformed = self.transform(image=np.array(image), mask=mask)
        else:
            transformed = ToTensorV2()(image=np.array(image), mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        return image_tensor, mask_tensor
