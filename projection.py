import numpy as np
import torch
import torch.nn.functional as F


class XrayProjector:
    def __init__(self, volume_shape=(128, 128, 128)):
        self.volume_shape = volume_shape

    def forward_project(self, volume, angle=0):
        """Equation 9"""
        is_numpy = isinstance(volume, np.ndarray)

        if is_numpy:
            volume = torch.from_numpy(volume).float()
            if volume.ndim == 3:
                volume = volume.unsqueeze(0).unsqueeze(0)

        if angle != 0:
            volume = self._rotate_volume(volume, angle)

        xray = volume.sum(dim=2) / (self.volume_shape[0] + 1e-8)

        if is_numpy:
            xray = xray.squeeze().numpy()

        return xray

    def back_project(self, xray, angle=0):
        """Equation 1"""
        is_numpy = isinstance(xray, np.ndarray)

        if is_numpy:
            xray = torch.from_numpy(xray).float()
            if xray.ndim == 2:
                xray = xray.unsqueeze(0).unsqueeze(0)

        B, C, H, W = xray.shape
        D = self.volume_shape[0]

        volume = xray.unsqueeze(2).repeat(1, 1, D, 1, 1)

        if angle != 0:
            volume = self._rotate_volume(volume, -angle)

        if is_numpy:
            volume = volume.squeeze().numpy()

        return volume

    def _rotate_volume(self, volume, angle):
        """Rotate around y-axis"""
        angle_rad = np.deg2rad(angle)

        theta = torch.tensor(
            [
                [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        grid = F.affine_grid(theta, volume.size(), align_corners=False)
        return F.grid_sample(volume, grid, align_corners=False)


def generate_drr_pairs(volume):
    projector = XrayProjector(volume.shape)
    ap_xray = projector.forward_project(volume, angle=0)
    lat_xray = projector.forward_project(volume, angle=90)
    return ap_xray, lat_xray
