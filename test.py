from Generator import Generator
from Dataset import MapDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import torch


dataset = MapDataset(r"C:\\Users\\KOUSHIK\\Desktop\\pix2pix\\val")
loader = DataLoader(dataset, batch_size=1)
model = Generator()

weights = torch.load(r"C:\Users\KOUSHIK\Desktop\pix2pix\Pix2Pix_Satellite_to_Map\gen.pth.tar",map_location=torch.device('cpu'))  

model.load_state_dict(weights["state_dict"])

inv_tra = T.Compose([T.Normalize(mean=[0, 0, 0], std=[1/0.5, 1/0.5, 1/0.5]),T.Normalize(mean=[-0.5,-0.5,-0.5],std=[1,1,1])])
transform = T.ToPILImage()

model.eval()

for idx, (i, t) in enumerate(loader):
    ans = model(i.float())
    ans = ans.squeeze(dim=0)
    ans = inv_tra(ans)
    ans = transform(ans)
    ans.save(f"outputs/{idx}.png")
