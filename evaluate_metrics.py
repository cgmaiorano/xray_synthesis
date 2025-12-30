import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json


def calculate_metrics(predicted, ground_truth):
    pred_norm = (predicted - predicted.min()) / (
        predicted.max() - predicted.min() + 1e-8
    )
    gt_norm = (ground_truth - ground_truth.min()) / (
        ground_truth.max() - ground_truth.min() + 1e-8
    )

    psnr = peak_signal_noise_ratio(gt_norm, pred_norm, data_range=1.0)
    ssim = structural_similarity(gt_norm, pred_norm, data_range=1.0)

    return psnr, ssim


def evaluate_results(results_dir):
    results_path = Path(results_dir)
    psnr_scores = []
    ssim_scores = []

    for npz_file in sorted(results_path.glob("example_*.npz")):
        data = np.load(npz_file)
        psnr, ssim = calculate_metrics(data["predicted_lat"], data["gt_lat"])
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        print(f"{npz_file.name}: PSNR={psnr:.2f} dB, SSIM={ssim:.3f}")

    results = {
        "mean_psnr": float(np.mean(psnr_scores)),
        "std_psnr": float(np.std(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "std_ssim": float(np.std(ssim_scores)),
    }

    print("\n" + "=" * 50)
    print(f"PSNR: {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
    print(f"SSIM: {results['mean_ssim']:.3f} ± {results['std_ssim']:.3f}")
    print("Target: PSNR > 17, SSIM > 0.7")
    print(
        f"Status: {'✓ PASS' if results['mean_psnr'] > 17 and results['mean_ssim'] > 0.7 else '✗ IMPROVE'}"
    )

    with open(results_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
