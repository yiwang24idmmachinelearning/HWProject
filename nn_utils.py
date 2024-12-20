import torch

from os import listdir, path

from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2

from image_utils import open_image

class SignDataset(Dataset):
  def __init__(self, imgs, labels, train_transform=None):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.imgs = imgs.to(self.device)
    self.labels = labels.to(self.device)
    self.train_transform = train_transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img, label = self.imgs[idx], self.labels[idx]
    if self.train_transform and torch.is_grad_enabled():
      img = self.train_transform(img)    
    return img, label

class SignUtils:
  img_transforms = v2.Compose([
    v2.Resize((224,224)),
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.RandomPerspective(distortion_scale=0.15, p=0.5),
  ])

  @staticmethod
  def train_test_split_cnn(test_ratio=0.5):
    train_ratio = 1.0 - test_ratio
    img_in_path = "./data/image/sign-language"
    fnames = sorted([f for f in listdir(img_in_path) if f.endswith("jpg")])

    pixels = []
    labels = []

    for f in fnames:
      label = f.split("-")[0]
      img = open_image(path.join(img_in_path, f))
      pixels.append(SignUtils.img_transforms(img))
      labels.append(label)

    SignUtils.i2l = sorted(list(set(labels)))
    SignUtils.l2i = {l:i for i,l in enumerate(SignUtils.i2l)}

    pixels_t = torch.stack(pixels)
    labels_t = torch.Tensor([SignUtils.l2i[l] for l in labels]).long()

    return random_split(SignDataset(pixels_t, labels_t, SignUtils.train_transforms), [train_ratio, test_ratio])
