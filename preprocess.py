import os
import numpy as np
import pydicom
from pathlib import Path
from scipy import ndimage
import json
from tqdm import tqdm


class CTPreprocessor:
    def __init__(self, target_spacing=(2.0, 2.0, 2.0), target_size=(128, 128, 128)):
        self.target_spacing = target_spacing
        self.target_size = target_size

    def load_dicom_series(self, dicom_dir):
        """Load and sort DICOM files"""
        dicom_files = []
        for f in Path(dicom_dir).glob("*.dcm"):
            try:
                dicom_files.append(pydicom.dcmread(str(f)))
            except Exception:
                continue

        if not dicom_files:
            return None, None

        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        volume = np.stack([d.pixel_array for d in dicom_files])
        intercept = dicom_files[0].RescaleIntercept
        slope = dicom_files[0].RescaleSlope
        volume = volume * slope + intercept

        pixel_spacing = dicom_files[0].PixelSpacing
        slice_thickness = dicom_files[0].SliceThickness
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

        return volume, spacing

    def normalize_hu(self, volume, window_center=-600, window_width=1500):
        """Normalize to [0,1] using lung window"""
        min_hu = window_center - window_width // 2
        max_hu = window_center + window_width // 2
        volume = np.clip(volume, min_hu, max_hu)
        return (volume - min_hu) / (max_hu - min_hu)

    def resample_volume(self, volume, original_spacing):
        """Resample to target spacing and size"""
        resize_factor = np.array(original_spacing) / np.array(self.target_spacing)
        volume_resampled = ndimage.zoom(volume, resize_factor, order=1)
        return self.crop_or_pad(volume_resampled, self.target_size)

    def crop_or_pad(self, volume, target_size):
        """Center crop or pad"""
        result = np.zeros(target_size, dtype=volume.dtype)

        starts_src = []
        starts_dst = []
        for i in range(3):
            if volume.shape[i] > target_size[i]:
                starts_src.append((volume.shape[i] - target_size[i]) // 2)
                starts_dst.append(0)
            else:
                starts_src.append(0)
                starts_dst.append((target_size[i] - volume.shape[i]) // 2)

        src_slices = tuple(
            slice(starts_src[i], starts_src[i] + min(volume.shape[i], target_size[i]))
            for i in range(3)
        )
        dst_slices = tuple(
            slice(starts_dst[i], starts_dst[i] + min(volume.shape[i], target_size[i]))
            for i in range(3)
        )

        result[dst_slices] = volume[src_slices]
        return result

    def process_volume(self, dicom_dir):
        """Complete preprocessing"""
        volume, spacing = self.load_dicom_series(dicom_dir)
        if volume is None:
            return None
        volume = self.normalize_hu(volume)
        volume = self.resample_volume(volume, spacing)
        return volume.astype(np.float32)


def preprocess_dataset(input_dir, output_dir, num_volumes=200):
    preprocessor = CTPreprocessor()
    os.makedirs(output_dir, exist_ok=True)

    patient_dirs = sorted([d for d in Path(input_dir).iterdir() if d.is_dir()])[
        :num_volumes
    ]

    successful = []
    for i, patient_dir in enumerate(tqdm(patient_dirs, desc="Processing")):
        try:
            volume = preprocessor.process_volume(patient_dir)
            if volume is not None:
                output_path = Path(output_dir) / f"volume_{i:03d}.npy"
                np.save(output_path, volume)
                successful.append(str(output_path))
        except Exception as e:
            print(f"Failed {patient_dir}: {e}")

    with open(Path(output_dir) / "manifest.json", "w") as f:
        json.dump({"volumes": successful}, f, indent=2)

    print(f"Processed {len(successful)}/{len(patient_dirs)} volumes")
