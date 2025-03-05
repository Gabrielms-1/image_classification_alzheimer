from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from config import Config

def get_albumentations_transform(resize):
    
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
    def __init__(self, root_dir, resize):
        self.root_dir = root_dir
        self.images, self.labels = self._get_images_path()
        
        self.resize = resize
        self.transform = get_albumentations_transform(resize)
        
        self.label_map_to_int = {label: i for i, label in enumerate(sorted(set(label for label in self.labels)))}
        self.int_to_label_map = {i: label for label, i in self.label_map_to_int.items()}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("L")
        image = np.array(image)
        label = (img_path).split("/")[-2]

        label = self.label_map_to_int[label]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, img_path
    

    def _get_images_path(self):
        all_items = glob.glob(os.path.join(self.root_dir, '**', '*.jpg'), recursive=True)
        labels = [item.split("/")[-2] for item in all_items]
        
        return all_items, labels
    
    def _get_transformations(self):
        transformations = transforms.Compose([
            transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.LANCZOS),  
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485], std=[0.229])  
            ])
        return transformations

def create_data_loaders(train_dataset, valid_dataset, batch_size):
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