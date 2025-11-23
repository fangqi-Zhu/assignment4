import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from nerf_dataset import NerfDataset
from nerf_model import NeRF

################################ IMPORTANT: This model is quite slow, you do not need to run it until it converges.  ###################################

# Position Encoding
class PositionalEncoder(nn.Module):
    """
    Implement the Position Encoding function.
    Defines a function that embeds x to (sin(2^k*pi*x), cos(2^k*pi*x), ...)
    Please note that the input tensor x should be normalized to the range [-1, 1].

    Args:
    x (torch.Tensor): The input tensor to be embedded.
    L (int): The number of levels to embed.

    Returns:
    torch.Tensor: The embedded tensor.
    """
    def __init__(self, data_range, L):
        super(PositionalEncoder, self).__init__()
        self.L = L
        self.data_range = data_range

    def forward(self, x):
        if self.data_range is not None:
            x = 2.0 * (x - self.data_range[0]) / (self.data_range[1] - self.data_range[0]) - 1.0
        device = x.device
        results = []
        for l in range(self.L):
            freq = (2 ** l) * np.pi
            for d in range(x.shape[-1]):
                results.append(torch.sin(freq * x[..., d][..., None]))
                results.append(torch.cos(freq * x[..., d][..., None]))
        pe = torch.cat(results, dim=-1)
        return pe


def sample_rays(H, W, f, c2w):
    """
    Samples rays from a camera with given height H, width W, focal length f, and camera-to-world matrix c2w.

    Args:
    H (int): The height of the image.
    W (int): The width of the image.
    f (float): The focal length of the camera.
    c2w (torch.Tensor): The 4x4 camera-to-world transformation matrix.

    Returns:
    rays_o (torch.Tensor): The origin of each ray, with shape (H, W, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (H, W, 3).
    """
    # Create a grid of pixel coordinates
    device = c2w.device
    i = torch.arange(0, W, dtype=torch.float32, device=device)
    j = torch.arange(0, H, dtype=torch.float32, device=device)
    x, y = torch.meshgrid(i, j, indexing='xy')  # (H, W)

    # Convert pixel coordinates to camera coordinates
    dirs = torch.stack((x - W / 2, -(y - H / 2), -f * torch.ones_like(x)), dim=-1)  # (H, W, 3)
    dirs = dirs / f  # normalize z to -1

    # Transform camera directions to world directions using the camera-to-world matrix
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)  # (H, W, 3)
    rays_d = F.normalize(rays_d, dim=-1)

    # The origin of each ray is the camera position
    rays_o = c2w[:3, -1].view(1, 1, 3).expand(H, W, -1)

    return rays_o, rays_d

def sample_points_along_the_ray(tn, tf, N_samples):
    """
    Samples points uniformly along a ray from time t_n to time t_f.

    Args:
    tn (torch.Tensor): The starting point of the ray.
    tf (torch.Tensor): The ending point of the ray.
    N_samples (int): The number of samples to take along the ray.

    Returns:
    torch.Tensor: A tensor of shape (N_rays, N_samples) containing the sampled t values along the ray.
    """
    device = tn.device
    N_rays = tn.shape[0]
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    t_vals = t_vals.expand([N_rays, -1])
    mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    upper = torch.cat([mids, torch.ones((N_rays, 1), device=device)], dim=-1)
    lower = torch.cat([torch.zeros((N_rays, 1), device=device), mids], dim=-1)
    t_rand = torch.rand(N_rays, N_samples, device=device)
    t_vals = lower + (upper - lower) * t_rand
    t_vals = tn[..., None] + t_vals * (tf[..., None] - tn[..., None])
    return t_vals

def volumn_render(NeRF, rays_o, rays_d, N_samples):
    """
    Performs volume rendering to generate an image from rays.

    Args:
    NeRF (nn.Module): The neural radiance field model.
    rays_o (torch.Tensor): The origin of each ray, with shape (N_rays, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (N_rays, 3).
    N_samples (int): The number of samples to take along each ray.

    Returns:
    torch.Tensor: The rendered RGB image.
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    tn = 2.0 * torch.ones(N_rays, device=device)
    tf = 6.0 * torch.ones(N_rays, device=device)
    
    # Sample points along each ray, from near plane to far plane
    t = sample_points_along_the_ray(tn, tf, N_samples)
    
    # Calculate the points along the rays by sampling
    # pts.shape => (N_rays, N_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t[..., None]
    
    # Get the color and density from the NeRF model
    viewdirs = rays_d[..., None, :].expand(-1, N_samples, -1)
    pts_flat = torch.reshape(pts, [-1, 3])
    viewdirs_flat = torch.reshape(viewdirs, [-1, 3])
    raw_rgb, raw_sigma = NeRF(pts_flat, viewdirs_flat)
    rgb = torch.reshape(raw_rgb, [N_rays, N_samples, 3])
    sigma = torch.reshape(raw_sigma, [N_rays, N_samples, 1])
    sigma = F.softplus(sigma)
    
    # Volume rendering: compute the transmittance and accumulate the color
    deltas = t[:, 1:] - t[:, :-1]  # (N_rays, N_samples-1)
    alphas = 1. - torch.exp(-sigma[:, :-1, :] * deltas.unsqueeze(-1))  # (N_rays, N_samples-1, 1)
    
    # Compute the weights for each sample using alpha compositing
    transmittance = torch.cumprod(
        torch.cat([torch.ones((N_rays, 1, 1), device=device), 1. - alphas + 1e-10], dim=1),
        dim=1
    )[:, :-1, :]  # (N_rays, N_samples-1, 1)
    weights = alphas * transmittance  # (N_rays, N_samples-1, 1)
    
    # Accumulate the color along each ray
    rgb_map = torch.sum(weights * rgb[:, :-1, :], dim=1)  # (N_rays, 3)
    
    return rgb_map
    


def random_select_rays(H, W, rays_o, rays_d, img, N_rand):
    """
    Randomly select N_rand rays to reduce memory usage.

    Parameters:
    - H: int, height of the image.
    - W: int, width of the image.
    - rays_o: torch.Tensor, original ray origins with shape (H * W, 3).
    - rays_d: torch.Tensor, ray directions with shape (H * W, 3).
    - img: torch.Tensor, image with shape (H * W, 3).
    - N_rand: int, number of random rays to select.

    Returns:
    - selected_rays_o: torch.Tensor, selected ray origins with shape (N_rand, 3).
    - selected_rays_d: torch.Tensor, selected ray directions with shape (N_rand, 3).
    - selected_img: torch.Tensor, selected image pixels with shape (N_rand, 3).
    """
    # Generate coordinates for all pixels in the image
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    
    # Randomly select N_rand indices without replacement
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
    
    # Select the corresponding coordinates, rays, and image pixels
    select_coords = coords[select_inds].long() # (N_rand, 2)
    selected_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = torch.tensor(selected_img, dtype=torch.float32)  # Ensure float32 dtype
    
    return selected_rays_o, selected_rays_d, selected_img


def fit_images_and_calculate_psnr(data_path, epochs=2000, learning_rate=5e-4):
    # get available device
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # 支持 macOS MPS 或 CPU
    device = 'cuda' if torch.cuda.is_available()  else 'cpu'  # 支持 macOS MPS 或 CPU
     
    # load data
    dataset = NerfDataset(data_path)

    # create model
    xyz_encoder = PositionalEncoder(data_range=[-4, 4], L=10).to(device)
    dir_encoder = PositionalEncoder(data_range=[-1, 1], L=4).to(device)
    nerf = NeRF(
        xyz_encoder=xyz_encoder,
        dir_encoder=dir_encoder,
        input_dim=60,
        view_dim=24,
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    # train the model
    N_samples = 64 # number of samples per ray
    N_rand = 1024 # number of rays per iteration, adjust according to your GPU memory
    for epoch in tqdm(range(epochs+1)):
        for i in range(len(dataset)):
            img, pose, focal = dataset[i]
            img = img.to(device)
            H, W = img.shape[:2]
            pose = pose.to(device)
            focal = focal.to(device)

            # sample rays
            rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

            # random select N_rand rays to reduce memory usage
            selected_rays_o, selected_rays_d, selected_gt_rgb = random_select_rays(H, W, rays_o, rays_d, img, N_rand)

            # volumn render
            pred_rgb = volumn_render(NeRF=nerf, rays_o=selected_rays_o, rays_d=selected_rays_d, N_samples=N_samples)

            l = loss(pred_rgb, selected_gt_rgb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            psnr_value = psnr(selected_gt_rgb.detach().cpu().numpy(), pred_rgb.detach().cpu().numpy(), data_range=1)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {l.item()}, PSNR: {psnr_value}')
            with torch.no_grad():
                chunk_size = 1024 # adjust according to your GPU memory
                pred_rgb = []
                for i in range(0, H*W, chunk_size):
                    rays_o_chunk = rays_o.reshape(-1, 3)[i:i+chunk_size]
                    rays_d_chunk = rays_d.reshape(-1, 3)[i:i+chunk_size]
                    pred_rgb.append(volumn_render(NeRF=nerf, rays_o=rays_o_chunk, rays_d=rays_d_chunk, N_samples=N_samples))
                pred_rgb = torch.cat(pred_rgb, dim=0)
                torchvision.utils.save_image(pred_rgb.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'output/NeRF/pred_{epoch}.png')
                torchvision.utils.save_image(img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'output/NeRF/gt_{epoch}.png')

                
if __name__ == '__main__':
    data_path = './data/lego' # data path
    psnr_value = fit_images_and_calculate_psnr(data_path)