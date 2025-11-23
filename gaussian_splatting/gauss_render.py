import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r, device):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r, device):
    '''
    Args:
    s (torch:Tensor): scale of gaussian with shape (N, 3)
    r (torch:Tensor): quaternion of gaussian with shape (N, 4)

    Using the provided build_rotation to get the rotation matrix from quaternion and build scale matrix for s.
    
    Return:
    (torch:Tensor) the matrix production of rotation matrix and scale matrix with shape (N, 3, 3) 
    '''
    R = build_rotation(r, device)
    S = torch.diag_embed(s)
    L = torch.bmm(R, S)
    return L


def strip_lowerdiag(L):
    device = L.device
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def corvariance_3d(s, r, device):
    '''
    Args:
        s (torch:Tensor): scale of gaussian with shape (N, 3)
        r (torch:Tensor): quaternion of gaussian with shape (N, 4)
        
        We use build_scaling_rotation to build matrix of R S and then we can obtain the covariance of 3D gaussian by RS(RS)^T
    
    Return
        (torch:Tensor)) 3D covariance with shape (N, 3, 3)
    '''
    L = build_scaling_rotation(s, r, device)
    cov3d = torch.bmm(L, L.transpose(-2, -1))
    return cov3d


def corvariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y, device
):
    """
    Compute the 2D image-space covariance for a batch of 3D Gaussians.

    Args:
        mean3d (torch.Tensor):     (N, 3)
            3D centers of Gaussians in world coordinates, one per Gaussian.

        cov3d (torch.Tensor):      (N, 3, 3)
            3D covariance matrices of Gaussians in world coordinates.

        viewmatrix (torch.Tensor): (4, 4)
            Camera extrinsic matrix (world-to-camera transform).
            The code uses viewmatrix[:3, :3] as rotation and viewmatrix[-1:, :3]
            as translation (note: last row stores translation in this convention).

        fov_x (float): horizontal field of view (radians).
        fov_y (float): vertical field of view (radians).
        focal_x (torch.Tensor): focal length in x, in pixel units.
        focal_y (torch.Tensor): focal length in y, in pixel units.

    Returns:
        cov2d (torch.Tensor): (N, 2, 2)
            2D covariance matrices in image (screen) space for each projected Gaussian,
            including a low-pass filter term for numerical stability / anti-aliasing.
    """
    
    # Precompute tangent of half FOVs for frustum clipping in camera space and transform 3D Gaussian centers from world space to camera space.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    # Project 3D centers into camera space using row-vector convention
    t_cam = homogeneous(mean3d) @ viewmatrix
    t = t_cam[..., :3]

    # Truncate Gaussians far outside the frustum.
    # We clip the normalized coordinates x/z, y/z, then re-scale by depth z.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Build Jacobian J of perspective projection to pixel space
    # Using truncated camera coords (tx, ty, tz)
    eps = 1e-6
    z = tz.clamp_min(eps)

    # Ensure fx, fy are tensors/scalars on the right device
    dtype = cov3d.dtype
    fx = focal_x if isinstance(focal_x, torch.Tensor) else torch.tensor(float(focal_x), device=device, dtype=dtype)
    fy = focal_y if isinstance(focal_y, torch.Tensor) else torch.tensor(float(focal_y), device=device, dtype=dtype)

    N = mean3d.shape[0]
    J = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * tx / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * ty / (z * z)

    # Transform world covariance to camera coordinates: C_cam = R^T C_world R
    # With row-vector convention, use W = R^T where R = viewmatrix[:3,:3]
    W = viewmatrix[:3, :3].T
    Wb = W[None, ...]
    C_cam = torch.bmm(Wb.expand_as(cov3d), torch.bmm(cov3d, Wb.transpose(1, 2).expand_as(cov3d)))

    # Project to 2D: cov2d = J C_cam J^T
    JC = torch.bmm(J, C_cam)
    cov2d = torch.bmm(JC, J.transpose(1, 2))
    
    # add low pass filter here according to E.q. 32 of EWQ splatting
    filter = torch.eye(2,2, device=device, dtype=dtype) * 0.3
    return cov2d + filter[None, :, :]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= -10
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh

class GaussRenderer(nn.Module):

    def __init__(self, active_sh_degree=3, white_bkgd=True, pixel_range=256, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pixel_range = pixel_range  # 改为参数
              
    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
    
    def render(self, camera, means2D, cov2d, color, opacity, depths, device):
        """
        Tile-based 2D Gaussian splatting renderer.

        Args:
            camera:
                Camera object providing image resolution:
                    - camera.image_width  (int)
                    - camera.image_height (int)
                self.pix_coord is assumed to be a precomputed (H, W, 2) tensor of pixel coords.

            means2D (torch.Tensor): (N, 2)
                Projected 2D centers of Gaussians in image space (pixel coordinates).

            cov2d (torch.Tensor): (N, 2, 2)
                2D image-space covariance matrices of Gaussians.

            color (torch.Tensor): (N, 3)
                RGB color for each Gaussian (in [0, 1]).

            opacity (torch.Tensor): (N, 1)
                Per-Gaussian base opacity (before per-pixel Gaussian weighting).

            depths (torch.Tensor): (N,)
                Per-Gaussian depth values (e.g., z in camera space),
                used for front-to-back sorting when compositing.
        Returns:
            dict with:
                - "render": (H, W, 3) final RGB image.
                - "alpha":  (H, W, 1) accumulated alpha map.
                - "visiility_filter": (N,) visibility mask (radii > 0).
                - "radii": (N,) per-Gaussian screen-space radius.
        """
        radii = get_radius(cov2d)
        rect_min, rect_max = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        # Initialize render buffers with background color
        bkgd = 1.0 if self.white_bkgd else 0.0
        self.render_color = torch.full((camera.image_height, camera.image_width, 3), bkgd, dtype=torch.float32, device=device)
        self.render_alpha = torch.zeros(camera.image_height, camera.image_width, 1, dtype=torch.float32, device=device)

        # 创建 pix_coord for this render
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(camera.image_height), torch.arange(camera.image_width), indexing='xy'), dim=-1).to(device)

        TILE_SIZE = 16
        for h in range(0, camera.image_height, TILE_SIZE):
            h_end = min(h + TILE_SIZE, camera.image_height)
            tile_h = h_end - h
            for w in range(0, camera.image_width, TILE_SIZE):
                w_end = min(w + TILE_SIZE, camera.image_width)
                tile_w = w_end - w
                # check if the rectangle penetrate the tile
                over_tl_x = torch.clamp(rect_min[:, 0], float(w), float(w_end - 1))
                over_tl_y = torch.clamp(rect_min[:, 1], float(h), float(h_end - 1))
                over_br_x = torch.clamp(rect_max[:, 0], float(w), float(w_end - 1))
                over_br_y = torch.clamp(rect_max[:, 1], float(h), float(h_end - 1))
                in_mask = (over_br_x > over_tl_x) & (over_br_y > over_tl_y) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue

                # Extract the pixel coordinates for this tile.
                tile_x = torch.arange(w, w_end, dtype=torch.float32, device=device)
                tile_y = torch.arange(h, h_end, dtype=torch.float32, device=device)
                tile_xx, tile_yy = torch.meshgrid(tile_x, tile_y, indexing='xy')
                tile_coord = torch.stack([tile_xx, tile_yy], dim=-1).reshape(-1, 2)  # (B, 2)
                B = tile_coord.shape[0]
        
                # Sort Gaussians by depth.
                indices = torch.argsort(depths[in_mask])  # near to far (ascending by default)
                gauss_in_tile = in_mask.nonzero().squeeze(-1)[indices]
                N_tile = gauss_in_tile.shape[0]
                sorted_means2D = means2D[gauss_in_tile]
                sorted_cov2d = cov2d[gauss_in_tile]
                sorted_color = color[gauss_in_tile]
                sorted_opacity = opacity[gauss_in_tile]
                sorted_depths = depths[gauss_in_tile]
        
                # Compute the distance from each pixel in the tile to the Gaussian centers.
                dx = tile_coord[:, None, :] - sorted_means2D[None, :, :]  # (B, N_tile, 2)
        
                # Compute the 2D Gaussian weight for each pixel.
                eye = torch.eye(2, device=device).unsqueeze(0)
                conics = torch.inverse(sorted_cov2d + 1e-6 * eye.expand(N_tile, -1, -1))
                quad = torch.einsum('bni,nij,bnj->bn', dx, conics, dx)
                gauss_weight = torch.exp(-0.5 * quad)  # (B, N_tile)
        
                # Compute the alpha blending using transmittance (T).
                alpha = gauss_weight.unsqueeze(-1) * sorted_opacity.unsqueeze(0)  # (B, N_tile, 1)
                ones_b = torch.ones((B, 1, 1), device=device)
                cumprod_input = torch.cat([ones_b, 1.0 - alpha + 1e-10], dim=1)
                T = torch.cumprod(cumprod_input, dim=1)[:, :-1]  # (B, N_tile, 1)
                weights = alpha * T  # (B, N_tile, 1)
        
                # Compute the color and depth contributions.
                tile_color = torch.sum(weights * sorted_color.unsqueeze(0), dim=1)  # (B, 3)
                tile_alpha = torch.sum(weights, dim=1)  # (B, 1)
        
                # Composite with background
                if self.white_bkgd:
                    tile_final = tile_color + (1.0 - tile_alpha) * 1.0
                else:
                    tile_final = tile_color + (1.0 - tile_alpha) * 0.0  # tile_color
        
                # Store computed values into rendering buffers.
                self.render_color[h:h_end, w:w_end] = tile_final.view(tile_h, tile_w, 3)
                self.render_alpha[h:h_end, w:w_end] = tile_alpha.view(tile_h, tile_w, 1)

        return {
            "render": self.render_color,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }

    def forward(self, camera, pc, device, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        
        mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                viewmatrix=camera.world_view_transform, 
                projmatrix=camera.projection_matrix)
        depths = mean_view[:, 2]
        
        color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        
        cov3d = corvariance_3d(scales, rotations, device)  # 传入 device
            
        cov2d = corvariance_2d(
            mean3d=means3D, 
            cov3d=cov3d, 
            viewmatrix=camera.world_view_transform,
            fov_x=camera.FoVx, 
            fov_y=camera.FoVy, 
            focal_x=camera.focal_x, 
            focal_y=camera.focal_y,
            device=device  # 传入 device
        )

        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
    
        # filter with in_mask to avoid invalid points
        valid_mask = in_mask & (radii > 0)  # but radii not yet, wait, compute radii first? Or just in_mask
        # For simplicity, filter with in_mask
        means2D = means2D[in_mask]
        cov2d = cov2d[in_mask]
        color = color[in_mask]
        opacity = opacity[in_mask]
        depths = depths[in_mask]
        
        rets = self.render(
            camera = camera, 
            means2D=means2D,
            cov2d=cov2d,
            color=color,
            opacity=opacity, 
            depths=depths,
            device=device
        )
        
        # No extra processing, background handled in render
        return rets
