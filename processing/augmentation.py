import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transform(height: int, width: int):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=0,
            p=0.3
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.3
        ),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])

def val_transform(height: int, width: int):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])