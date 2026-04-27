from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from src.config import config

class FoodDataset(Dataset):
    '''
    Custom Dataset for image classification
    Expects dataframe with columns:
    -image_path
    -label(optional for testing data)
    '''

    def __init__(self,dataframe, transform = None,is_test = False):
        self.dataframe = dataframe
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
    
    
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["image_path"].name
        label = int(row["label"])
        return image, label



def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_valid_transforms():
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_dataloaders(df, valid_size = 0.2):
    """
    Splits dataframe into train/validation and returns dataloaders.
    """

    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        stratify=df["label"],
        random_state=config.seed
    )

    train_dataset = FoodDataset(
        train_df,
        transform=get_train_transforms()
    )

    valid_dataset = FoodDataset(
        valid_df,
        transform=get_valid_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, valid_loader


def create_test_loader(test_df):
    test_dataset = FoodDataset(
        test_df,
        transform=get_valid_transforms(),
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return test_loader