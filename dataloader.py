import os
from PIL import Image
from torch.utils.data import Dataset

class NoClassImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        # List all files in the root_dir and filter for common image extensions
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                self.image_files.append(os.path.join(root_dir, filename))
        if not self.image_files:
            raise RuntimeError(f"Found 0 image files in {root_dir}. Please ensure the directory contains images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') # Ensure 3 channels

        if self.transform:
            image = self.transform(image)
        return image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations for the images
hin_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor() # Convert PIL Image to PyTorch Tensor
])

# Load the dataset from a folder (assuming images are in a folder named 'data' in the content directory)
# You might need to upload your image folder to /content/data or specify the correct path
dataset = NoClassImageDataset("C:\\Datasets\\faces\\resized", hin_transforms)

# Create a DataLoader
batch_size = 64 # You can adjust this batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)