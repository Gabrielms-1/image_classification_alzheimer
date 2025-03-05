from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from config import Config

def get_albumentations_transform(resize: int) -> A.Compose:
    """
    Create an Albumentations transformation pipeline for image preprocessing.

    Parameters:
        resize (int): The target size to resize the image (both width and height).

    Returns:
        A.Compose: The composed Albumentations transformation pipeline.
    """
    params = Config.AUGMENTATION_TRANSFORMATIONS

    return A.Compose([
        A.Resize(height=resize, width=resize, interpolation=cv2.INTER_LANCZOS4),
        #A.ToGray(p=1.0), 
        A.Normalize(mean=[0.485], std=[0.229]),
        #A.RandomBrightnessContrast(brightness_limit=params["brightness_limit"], contrast_limit=params["contrast_limit"], p=params["p"]),
        A.ShiftScaleRotate(shift_limit=params["shift_limit"], scale_limit=params["scale_limit"], rotate_limit=params["rotate_limit"],
                       border_mode=cv2.BORDER_CONSTANT, p=params["p"]),
        #A.ElasticTransform(alpha=params["alpha"], sigma=params["sigma"], alpha_affine=params["alpha_affine"], interpolation=cv2.INTER_LANCZOS4, p=params["p"]),
        ToTensorV2()
    ])

class FolderBasedDataset(Dataset):
    """
    A custom dataset class that loads images from folder organized data.

    Attributes:
        root_dir (str): Root directory containing image folders.
        images (list): List of image file paths.
        labels (list): List of labels corresponding to image folders.
        resize (int): Target size for image resizing.
        transform (callable): Transformation pipeline applied on the images.
        label_map_to_int (dict): Mapping from class labels to integer indices.
        int_to_label_map (dict): Mapping from integer indices to class labels.
    """

    def __init__(self, root_dir: str, resize: int):
        """
        Initialize the FolderBasedDataset.

        Parameters:
            root_dir (str): Root directory containing the dataset.
            resize (int): The target size to resize the images.
        """
        self.root_dir = root_dir
        self.images, self.labels = self._get_images_path()
        
        self.resize = resize
        self.transform = get_albumentations_transform(resize)
        
        self.label_map_to_int = {label: i for i, label in enumerate(sorted(set(label for label in self.labels)))}
        self.int_to_label_map = {i: label for label, i in self.label_map_to_int.items()}

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retrieve an image and its label by index.

        Parameters:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: (image tensor, integer label, image file path)
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert("L")
        image = np.array(image)
        label = (img_path).split("/")[-2]
        label = self.label_map_to_int[label]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, img_path
    
    def _get_images_path(self):
        """
        Retrieve image file paths and corresponding labels from the dataset directory.

        Returns:
            tuple: (list of image paths, list of labels)
        """
        all_items = glob.glob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True)
        labels = [item.split("/")[-2] for item in all_items]
        return all_items, labels


def create_data_loaders(train_dataset: Dataset, valid_dataset: Dataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation datasets.

    Parameters:
        train_dataset (Dataset): The training dataset.
        valid_dataset (Dataset): The validation dataset.
        batch_size (int): Batch size for the data loaders.

    Returns:
        tuple: (train DataLoader, validation DataLoader)
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=False)
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False)
    
    return train_loader, valid_loader

if __name__ == "__main__":
    train_dataset = FolderBasedDataset(root_dir="data/raw/train", resize=224)
    valid_dataset = FolderBasedDataset(root_dir="data/raw/valid", resize=224)
    
    _, valid_loader = create_data_loaders(train_dataset, valid_dataset, batch_size=16)
    
    valid_loader