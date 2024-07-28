import os
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
import constants
import albumentations as A

from typing import List
from enum import StrEnum


class Mode(StrEnum):
    TRAIN = "Train"
    TEST = "TEST"


class SateImageDataset(Dataset):
    def __init__(self, K: List[int], mode: Mode, edge_detection=False):

        self.root_dir = "Dataset"
        self.image_dir = "Images"
        self.mask_dir = "Labels"

        self.K = K
        self.mode = mode
        self.edge_detection = edge_detection

        if self.mode == Mode.TRAIN:
            self.transform = A.Compose([
                A.RandomCrop(width=448, height=448),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # TEST
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.images = []
        self.masks = []

        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv.imread(self.masks[idx]), cv2.COLOR_BGR2RGB)
        mask = self._one_hot_encoding(mask, constants.CLASS_RGB_VALUES).astype("float")

        transformed = self.transform(image=image, mask=mask)

        image = transforms.ToTensor()(transformed["image"])
        mask = transforms.ToTensor()(transformed["mask"])

        # Image -> [channel=3, width, height]
        # Mask => [channel=2, width, height]
        image = image.type(torch.float)
        mask = mask.type(torch.float)

        if self.edge_detection:
            # Image -> [channel=4, width, height]
            edges = self.__image_edge_detection(transformed["image"].astype(np.uint8))
            image = torch.concat([image, edges], dim=0)

        return image, mask

    @staticmethod
    def __image_edge_detection(ndimage):
        image = cv2.cvtColor(ndimage, cv.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, 30, 200)
        return transforms.ToTensor()(edges)


    @staticmethod
    def _one_hot_encoding(mask: np.ndarray, rgb_values: List[List[int]] = constants.CLASS_RGB_VALUES):
        class_map = []
        for color in rgb_values:
            eq = np.equal(mask, color)
            class_map.append(np.all(eq, axis=-1))

        return np.stack(class_map, axis=-1)

    def _load_data(self):
        for k in self.K:
            for file_name in os.listdir(os.path.join(self.root_dir, str(k), self.image_dir)):
                self.images.append(os.path.join(self.root_dir, str(k), self.image_dir, file_name))
                self.masks.append(os.path.join(self.root_dir, str(k), self.mask_dir, file_name))


def get_datasets(test_k: int, edge_detection=False):
    options = [1, 2, 3, 4, 5]

    if test_k not in options:
        raise Exception("test_k must in 1, 2, 3, 4, 5")

    options.remove(test_k)

    test_ds = SateImageDataset(K=[test_k], mode=Mode.TEST, edge_detection=edge_detection)
    train_ds = SateImageDataset(K=options, mode=Mode.TRAIN, edge_detection=edge_detection)

    train_ds, val_ds = random_split(train_ds, [0.9, 0.1])

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    train_ds, val_ds, test_ds = get_datasets(1, edge_detection=True)

    image, mask = test_ds.__getitem__(1)

    _, ax = plt.subplots(1, 4)

    ax[0].imshow(transforms.ToPILImage()(inverse_normalize(image[:3])))
    ax[1].imshow(transforms.ToPILImage()(image[3]))
    ax[2].imshow(transforms.ToPILImage()(image[:3]))
    ax[3].imshow(transforms.ToPILImage()(mask.argmax(dim=0).float()))

    plt.show()


