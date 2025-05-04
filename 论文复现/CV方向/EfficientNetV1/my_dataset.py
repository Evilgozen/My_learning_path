from PIL import Image
import torch
from pydantic.experimental.pipeline import transform
from torch.utils.data import Dataset

class MyDataSet(Dataset):

    def __init__(self,images_path:list,images_class:list,transform=None):
        self.images_path = images_path #图像文件路径
        self.images_class = images_class #标签列表
        self.transform = transform #图像预处理/增强变换

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img=Image.open(self.images_path[item])

        if img.mode != 'RGB':
            raise ValueError(f"image: {self.images_path[item]} isn't RGB ")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img,label

    @staticmethod
    def collate_fn(batch): #将batch中的元素整合成一个stack，用来dataloader
        images,labels = tuple(zip(*batch))

        images = torch.stack(images,dim=0)
        labels = torch.as_tensor(labels)
        return images,labels