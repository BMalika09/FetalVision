import pandas as pd
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import Dataset





class ResizeWithPadding:
    """Resize a PIL image without changing its aspect ratio, then pad"""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(
            image,
            size=(self.size, self.size),
            method=Image.Resampling.BILINEAR,
            color=0,
            centering=(0.5, 0.5),
        )


def build_transforms(train=False, size=224):
    transforms = [
        v2.Grayscale(num_output_channels=3),
        ResizeWithPadding(size),
    ]

    if train:
        transforms.extend([
            v2.RandomRotation(
                degrees=5,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            v2.ColorJitter(brightness=0.10, contrast=0.10),
        ])

    transforms.extend([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return v2.Compose(transforms)



CLASSES = [
    "Fetal abdomen",
    "Fetal brain",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other",
]

CLASS_TO_IDX = {
    name: index for index, name in enumerate(CLASSES)
}



class FetalPlanesDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = row["Image_name"]
        image_path = self.images_dir / f"{image_id}.png"

        with Image.open(image_path) as image:
            image = image.copy()

        image = self.transform(image)
        label = CLASS_TO_IDX[row["Plane"]]

        return {
            "image": image,
            "label": label,
            "image_id": image_id,
            "patient_id": int(row["Patient_num"]),
        }