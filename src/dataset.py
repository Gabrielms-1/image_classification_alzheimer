from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os

class FolderBasedDataset(Dataset):
    def __init__(self, root_dir, resize):
        self.root_dir = root_dir
        self.images, self.labels = self.get_images_path()
        
        self.resize = resize
        self.transform = self.get_transformations()
        
        self.label_map_to_int = {label: i for i, label in enumerate(sorted(set(label for label in self.labels)))}
        self.int_to_label_map = {i: label for label, i in self.label_map_to_int.items()}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = (img_path).split("/")[-2]

        label = self.label_map_to_int[label]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
    

    def get_images_path(self):
        all_items = glob.glob(os.path.join(self.root_dir, '**', '*', '*.jpg'), recursive=True)
        labels = [item.split("/")[-2] for item in all_items]
        
        return all_items, labels
    
    def get_transformations(self):
        transformations = transforms.Compose([
            transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.LANCZOS),  
            transforms.ToTensor(),  
            # transforms.Grayscale(num_output_channels=1),       
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])  
            ])
        return transformations
    

if __name__ == "__main__":
    dataset = FolderBasedDataset(root_dir="data/raw", resize=224)

    image, label, path = dataset[0]
    dataset