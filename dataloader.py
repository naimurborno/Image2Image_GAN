
import numpy as np
import torch
import random
from os import listdir
from os.path import join, isfile
from torch.utils.data import Dataset
from train_config

config=train_config.config


def norm01(x, eps=1e-8):
    x = x - x.min()
    return x / (x.max() + eps)
def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x))
def ifft2c(F):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))

def make_circular_lowpass_mask(H, W, radius):
    """
    radius: in pixels (0..min(H,W)/2). Larger radius keeps more low-frequency.
    """
    cy, cx = H // 2, W // 2
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    dist2 = (y - cy)**2 + (x - cx)**2
    return (dist2 <= radius**2).astype(np.float32)
# CT dataset
class CT_Dataset(Dataset):
  def __init__(self, path_full, path_quarter, transform):
    # Path of 'full_dose' folders
    self.path_full = path_full
    self.path_quarter=path_quarter
    self.transform = transform
    self.file_full = list()
    self.file_quarter=list()
    for subdir_name in sorted(listdir(self.path_full)):
      # print(subdir_name)
      subdir_path = join(self.path_full, subdir_name)  # Get the full path to the subdirectory
      self.file_full.append(subdir_path)
      self.file_quarter.append(join(self.path_quarter,subdir_name))

  def __len__(self):
    return len(self.file_full)

  def __getitem__(self, idx):
    x_F = np.load(self.file_full[idx])
    x_Q=np.load(self.file_quarter[idx])

    # Convert to HU scale
    x_F = (x_F - 0.0192) / 0.0192 * 1000
    x_Q = (x_Q - 0.0192) / 0.0192 * 1000

    # Normalize images
    x_F[x_F < -1000] = -1000
    x_Q[x_Q < -1000] = -1000
    x_F = x_F / 4000 + config["offset"]
    x_Q = x_Q / 4000 + config["offset"]

    # Apply transform
    x_F = self.transform(x_F)
    x_Q=self.transform(x_Q)
    # print(x_F.shape)
    _,H,W=x_Q.shape
    mask_low=make_circular_lowpass_mask(H,W,config["radius"])
    mask_high=1.0-mask_low

    F_F=fft2c(x_F)
    F_Q=fft2c(x_Q)

    low_F=norm01(ifft2c(F_F*mask_low),config["eps"])
    high_F=norm01(ifft2c(F_F*mask_high), config["eps"])

    low_Q=norm01(ifft2c(F_Q*mask_low),config["eps"])
    high_Q=norm01(ifft2c(F_Q*mask_high), config["eps"])

    file_name = self.file_full[idx]
    return x_Q, x_F, mask_low,mask_high, file_name
