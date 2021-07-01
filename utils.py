
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


def PointNetLoss(outputs: torch.Tensor, labels: torch.Tensor, M3: torch.Tensor, M64: torch.Tensor, alpha=1e-3):
    criterion = nn.CrossEntropyLoss()
    batch_size = outputs.size(0)
    I1 = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1)
    I2 = torch.eye(64, requires_grad=True).repeat(batch_size, 1, 1)
    if outputs.is_cuda:
        I1 = I1.cuda()
        I2 = I2.cuda()
    DIFF1 = I1-torch.bmm(M3, M3.transpose(1, 2))
    DIFF2 = I2-torch.bmm(M64, M64.transpose(1, 2))
    cls_loss = criterion(outputs, labels)
    mat_loss = alpha*(torch.norm(DIFF1)+torch.norm(DIFF2))/batch_size
    return cls_loss + mat_loss


class PointNetDataset(Dataset):
    def __init__(self, x, y, train: bool = True, random_seed=2021) -> None:
        super().__init__()
        np.random.seed(random_seed)
        np.random.shuffle(x)
        np.random.seed(random_seed)
        np.random.shuffle(y)
        if train:
            self.transform = transforms.Compose([
                Normalize(),
                Rotate(),
                Jitter(),
                ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                Normalize(),
                ToTensor(),
            ])
        self.data = x
        self.label = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return len(self.data)


class Normalize(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return norm_pointcloud


class Rotate(object):
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        angle = np.random.uniform()*2*math.pi
        rotation_matrix = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])
        rotated_pointcloud = np.dot(rotation_matrix, pointcloud.T).T
        return rotated_pointcloud


class Jitter(object):
    def __init__(self, sigma=0.01, clip=0.05) -> None:
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        noise = np.clip(self.sigma*np.random.randn(*
                        pointcloud.shape), -self.clip, self.clip)
        return pointcloud+noise


class ToTensor(object):
    def __call__(self, pointcloud: np.ndarray) -> torch.Tensor:
        return torch.Tensor(pointcloud)
