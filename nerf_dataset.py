import os
import torch
import json
import cv2
import imageio
import numpy as np

class NerfDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and processing data for NeRF (Neural Radiance Fields) model training.
    
    Attributes:
        root_path (str): The root directory path where the dataset is stored.
    """

    def __init__(self, root_path, split='train', downsample=8):
        super().__init__()
        self.root_path = root_path
        self.split = split
        self.downsample = downsample

        with open(f'{root_path}/transforms_{split}.json', 'r') as f:
            meta = json.load(f)

        imgs = []
        poses = []
        for frame in meta['frames']:
            image = imageio.imread(os.path.join(root_path, frame['file_path'] + '.png'))
            image = cv2.resize(image, (image.shape[1] // downsample, image.shape[0] // downsample))
            imgs.append(image)
            poses.append(np.array(frame['transform_matrix']))

        #normmlization and add white background
        imgs = np.array(imgs) / 255.
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
        poses = np.array(poses)
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.poses = torch.tensor(poses, dtype=torch.float32)

        # all images are of the same size
        self.H, self.W = imgs[0].shape[:2]
        self.camera_angle_x = torch.tensor(float(meta['camera_angle_x']))
        self.focal = torch.tensor(0.5 * self.W / np.tan(0.5 * self.camera_angle_x))

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return self.imgs.shape[0]

    @torch.no_grad()
    def __getitem__(self, index):
        """
        Returns a sample from the dataset, which includes image, pose, and camera focal length.
        """
        image = self.imgs[index]
        pose = self.poses[index]
        focal = self.focal

        return image, pose, focal