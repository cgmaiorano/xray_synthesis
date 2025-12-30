import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from projection import XrayProjector, generate_drr_pairs


class XraySynthesisDataset(Dataset):
    def __init__(self, volume_dir, split="train", train_ratio=0.8):
        self.projector = XrayProjector()

        with open(Path(volume_dir) / "manifest.json", "r") as f:
            manifest = json.load(f)

        volumes = manifest["volumes"]
        split_idx = int(len(volumes) * train_ratio)

        self.volume_paths = (
            volumes[:split_idx] if split == "train" else volumes[split_idx:]
        )

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        volume = np.load(self.volume_paths[idx])
        ap_xray, lat_xray = generate_drr_pairs(volume)
        back_proj_volume = self.projector.back_project(ap_xray, angle=0)

        return {
            "input": torch.from_numpy(back_proj_volume).float().unsqueeze(0),
            "target": torch.from_numpy(volume).float().unsqueeze(0),
            "lat_gt": torch.from_numpy(lat_xray).float(),
        }
