import torch
import numpy as np
import torchvision
import random
import os

from tqdm import tqdm
from plyfile import PlyData, PlyElement
from skimage.metrics import peak_signal_noise_ratio as psnr

from nerf_dataset import NerfDataset
from gaussian_splatting.gauss_render import GaussRenderer
from gaussian_splatting.utils.camera_utils import Camera
from gaussian_splatting.gauss_model import GSModel
from gaussian_splatting.utils.point_utils import PointCloud
import gaussian_splatting.utils.loss_utils as loss_utils

class GSSTrainer():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(pixel_range=100, white_bkgd=True)  # Pass white_bkgd
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0

    def get_intrinsic_torch(self, h, w, focal, device='cpu'):
        cx = w / 2.0
        cy = h / 2.0
        
        intrinsic = torch.tensor([
            [focal, 0,     cx,    0],
            [0,     focal, cy,    0],
            [0,     0,     1,     0],
            [0,     0,     0,     1]
        ], dtype=torch.float32, device=device)
        
        return intrinsic
    def train(self, data_path, epochs=200, learning_rate=1e-3, use_ply=False, ply_path=None):

        # Support CUDA, MPS, and CPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        # model definition
        gaussModel = GSModel(sh_degree=4, debug=False).to(device)  # Move to device

        # Decide whether to use fetchPly based on use_ply parameter
        if use_ply and ply_path is not None:
            # load from ply file
            xyz, shs = fetchPly(ply_path)
            num_pts = len(xyz)
            print(f"Loaded {num_pts} points from PLY file: {ply_path}")
        else:
            # randomly generate some points
            # you can adjust num_pts according to your GPU memory
            # larger num_pts will give better quality but require more memory and time
            num_pts = 2**14
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
        
        channels = dict(
            R=shs[..., 0],
            G=shs[..., 1],
            B=shs[..., 2],
        )
        pointcloud = PointCloud(xyz, channels)
        raw_points = pointcloud.random_sample(num_pts)
        gaussModel.create_from_pcd(raw_points, device=device)  # Pass device

        optimizer = torch.optim.Adam(gaussModel.parameters(), lr=learning_rate)

        # load data
        dataset = NerfDataset(data_path)
        dataset_test = NerfDataset(data_path, split='test')
        
        # Decide output path based on use_ply parameter
        if use_ply:
            output_dir = 'output/3DGS_ply'
        else:
            output_dir = 'output/3DGS'
        os.makedirs(output_dir, exist_ok=True)

        for epoch in tqdm(range(epochs+1)):
            for i in tqdm(range(len(dataset)), desc=f"Training Epoch {epoch}"):
                img, pose, focal = dataset[i]
                img = img.to(device)
                pose = pose.to(device)
                pose[:3, 1:3] *= -1
                focal = focal.to(device)
                H, W = img.shape[:2]
                intrinsics = self.get_intrinsic_torch(H, W, focal, device=device)
                camera = Camera(width=W, height=H, intrinsic=intrinsics, c2w=pose)
                out = self.gaussRender(camera=camera, pc=gaussModel, device=device)  # Pass device

                l1_loss = loss_utils.l1_loss(out['render'], img)
                ssim_loss = 1.0-loss_utils.ssim(out['render'], img)

                total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss 
                psnr_value = psnr(img.detach().cpu().numpy(), out['render'].detach().cpu().numpy(), data_range=1)
                
                print(f"Epoch {epoch}, Step {i}: Loss {total_loss.item():.4f}, PSNR {psnr_value:.2f}")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                # print(log_dict)
                print("Evaluating...")
                psnr_value_test = 0.0
                for i_test in tqdm(range(len(dataset_test)), desc=f"Testing Epoch {epoch}"):
                    img_test, pose_test, focal_test = dataset_test[i_test]
                    img_test = img_test.to(device)
                    pose_test = pose_test.to(device)
                    pose_test[:3, 1:3] *= -1
                    focal_test = focal_test.to(device)
                    intrinsics_test = self.get_intrinsic_torch(H, W, focal_test, device=device)
                    camera_test = Camera(width=W, height=H, intrinsic=intrinsics_test, c2w=pose_test)
                    out_test = self.gaussRender(camera=camera_test, pc=gaussModel, device=device)  # Pass device
                    psnr_value_test += psnr(img_test.detach().cpu().numpy(), out_test['render'].detach().cpu().numpy(), data_range=1)
                    if i_test  == 29:
                        torchvision.utils.save_image(out_test['render'].reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'{output_dir}/pred_{epoch}.png')
                        torchvision.utils.save_image(img_test.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'{output_dir}/gt_{epoch}.png')

                print(f'Test PSNR for epoch {epoch}: {psnr_value_test/len(dataset_test):.2f}')


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return positions, colors

if __name__ == '__main__':

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    data_path = './data/lego' # data path
    
    # Set whether to use PLY file
    use_ply = True  # Set to True to use PLY file
    ply_path = 'data/lego/fused_light.ply'  # PLY file path
    
    trainer = GSSTrainer()
    trainer.train(data_path, use_ply=use_ply, ply_path=ply_path)
