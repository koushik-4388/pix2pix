import numpy as np
import Config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
transform = T.Resize((256,256))
tra = T.Compose([T.PILToTensor(),T.Resize((256,512))])

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]
        
        augmentations = Config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]
        print(target_image.shape)
        input_image = Config.transform_only_input(image=input_image)["image"]
        target_image = Config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset(r"C:\Users\KOUSHIK\Desktop\pix2pix\train")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(y.shape)
        save_image(x.float(), "x.png")
        save_image(y.float(), "y.png")
        import sys

        sys.exit()
