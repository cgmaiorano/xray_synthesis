import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from model import UNet3D
from dataset import XraySynthesisDataset
from projection import XrayProjector


def synthesize_lateral_view(model, ap_xray, device):
    projector = XrayProjector()
    back_proj_volume = projector.back_project(ap_xray, angle=0)

    input_tensor = (
        torch.from_numpy(back_proj_volume).float().unsqueeze(0).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        predicted_ct = model(input_tensor)

    predicted_ct = predicted_ct.squeeze().cpu().numpy()
    predicted_lat = projector.forward_project(predicted_ct, angle=90)

    return predicted_lat, predicted_ct


def generate_examples(model_path, data_dir, output_dir, num_examples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = XraySynthesisDataset(data_dir, split="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for i, batch in enumerate(test_loader):
        if i >= num_examples:
            break

        back_proj = batch["input"].squeeze().cpu().numpy()
        ap_xray = back_proj[0, :, :]
        lat_gt = batch["lat_gt"].squeeze().cpu().numpy()

        predicted_lat, predicted_ct = synthesize_lateral_view(model, ap_xray, device)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ap_xray, cmap="gray")
        axes[0].set_title("Input: AP X-ray (0°)")
        axes[0].axis("off")

        axes[1].imshow(predicted_lat, cmap="gray")
        axes[1].set_title("Predicted: Lateral X-ray (90°)")
        axes[1].axis("off")

        axes[2].imshow(lat_gt, cmap="gray")
        axes[2].set_title("Ground Truth: Lateral X-ray")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path / f"example_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

        np.savez(
            output_path / f"example_{i:02d}.npz",
            ap=ap_xray,
            predicted_lat=predicted_lat,
            gt_lat=lat_gt,
            predicted_ct=predicted_ct,
        )

    print(f"Generated {num_examples} examples in {output_dir}")
