
import os
import h5py
import numpy as np
import pandas as pd
import logging
import argparse
from PIL import Image
import sigpy as sp
from sigpy.mri import app

import torch
from torchvision import transforms

from metrics.ssfd import SSFD
from metrics.lpip import LPIPS
from metrics.rinfd import RINFD
from metrics.dists import DISTS
from metrics.hfen import HFEN
from metrics.vif import VIF
from metrics.nqm import NQM
from meddlr.metrics.image import SSIM, PSNR, NRMSE

logger = logging.getLogger(__name__)


def compute_covariance(kspace, corner_fraction=0.05):
    """
    Compute the inter-coil covariance matrix from the corners of  k-space.

    Args:
        kspace (np.ndarray): Complex k-space data with shape (nx, ny, ncoils).
        corner_fraction (float): Fraction of k-space corners used to compute noise covariance.

    Returns:
        np.ndarray: Inter-coil covariance matrix with shape (ncoils, ncoils).
    """

    nx, ny, ncoils = kspace.shape
    
    # Calculate corner sizes
    corner_size_x = int(round(corner_fraction * nx))
    corner_size_y = int(round(corner_fraction * ny))

    # Extract corners (noise-only regions)
    corners = []
   
    corners.extend([
        kspace[:corner_size_x, :corner_size_y, :].reshape(-1, ncoils),
        kspace[:corner_size_x, -corner_size_y:, :].reshape(-1, ncoils),
        kspace[-corner_size_x:, :corner_size_y, :].reshape(-1, ncoils),
        kspace[-corner_size_x:, -corner_size_y:, :].reshape(-1, ncoils),
    ])

    # Concatenate corners
    corners = np.concatenate(corners, axis=0)

    # Compute inter-coil covariance matrix
    cov = np.cov(corners.T)
    return cov

def generate_complex_noise(cov, size):
    """
    Generate complex noise following a specified complex covariance matrix.
    
    Args:
        cov (np.ndarray): Complex covariance matrix (ncoils x ncoils).
        size (np.ndarray): Dimension of noise to generate.
    
    Returns:
        np.ndarray: Complex noise samples (num_samples x N).
    """
    # Separate the real and imaginary parts
    cov_real = cov.real
    cov_imag = cov.imag
    
    # Construct the block covariance matrix
    cov_block = np.block([
        [cov_real, -cov_imag],
        [cov_imag,  cov_real]
    ])
    
    # Generate real-valued multivariate normal samples
    mean = np.zeros(2 * cov.shape[0])  # Zero-mean
    noise = np.random.multivariate_normal(mean, cov_block, size=size)
    
    # Split into real and imaginary parts
    half_size = cov.shape[0]
    real_part = noise[:,:, :half_size]
    imag_part = noise[:,:, half_size:]
    
    # Combine into complex samples
    complex_noise = real_part + 1j * imag_part

    return complex_noise

def add_noise_with_scaled_covariance(kspace, cov, noise_increase=0.1):
    """
    Add proportional noise to k-space data, accounting for existing noise.

    Args:
        kspace (np.ndarray): Original complex k-space data with shape (H, W, coils).
        cov (np.ndarray): Inter-coil covariance matrix (2*coils, 2*coils).
        noise_increase (float): Fractional increase in total noise variance.

    Returns:
        np.ndarray: Noisy k-space data with the same shape as the input.
    """
    nkx, nky, ncoils = kspace.shape
    noise = noise_increase * generate_complex_noise(cov, size=(nkx, nky)) 
    kspace_noisy = kspace + noise

    return kspace_noisy

def load_image(image_path):
    # Load an image and convert it to a tensor
    image = Image.open(image_path).convert('L') 
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) 
    return image_tensor

def process_kspace_slice(kspace, calib_size=24, device=0, nmaps=1):
    """Process 2D multi-coil kspace slice with ESPIRIT.
    
    Args:
        kspace (np.ndarray): Shape (H, W, coils)
        calib_method (str): Calibration method
        calib_size (int): Calibration size
        device (int): Device ID (-1 for CPU)
        nmaps (int): Number of maps
    
    Returns:
        tuple: (image, maps)
    """
    
    if device == -1:
        device = sp.cpu_device
    else:
        device = sp.Device(device)
    
    
    maps = app.EspiritCalib(
        kspace,
        calib_width=calib_size,
        device=device,
        show_pbar=False,
        crop=0.1,
        kernel_width=6,
        max_iter=100,
        thresh=0.05,
    ).run()
    
    if not isinstance(maps, np.ndarray):
        maps = sp.to_device(maps, sp.cpu_device)
    
    lamda = 0.01
    image = app.SenseRecon(kspace, maps, lamda=lamda).run()

    return image, maps

def compute_metrics_acq_noise(img_dir, kspace_dir, results_dir, noise_increase_levels, device=0):
    """Compute metrics comparing reconstructions against ground truth with covariance scaled complex k-space noise.
    
    Args:
        img_dir (str): Directory with PNG images of reconstructions.
        kspace_dir (str): Directory with k-space HDF5 files.
        results_dir (str): Output directory for results CSV file.
        noise_increase_levels (list of float): List of covariance matrix scale factors to add noise to k-space data.
        device (int): Device ID for processing (0 for GPU, -1 for CPU).
        
    """

    metrics_dict = {
        "SSIM": SSIM(im_type=None),
        "PSNR": PSNR(im_type=None),
        "NRMSE": NRMSE(im_type=None),
        "HFEN": HFEN(sigma=1.5),
        "VIF": VIF(),
        "NQM": NQM(),
        "SSFD": SSFD(),
        "LPIPS (VGG-16)": LPIPS(net_type='vgg', lpips=True,),
        "ResNet50 (RadImageNet)": RINFD(model_weights_mode="RadImageNet"),
    }

    results_csv = os.path.join(results_dir, "metrics_acq_noise.csv")
    os.makedirs(results_dir, exist_ok=True)
    results = []

    methods = ["UNet", "Unrolled"]
    accelerations = [2, 4, 6]

    for method in methods:
        for acc in accelerations:
            folder_name = f"{method}_{acc}x"
            recon_folder = os.path.join(img_dir, folder_name)

            for recon_file in sorted(os.listdir(recon_folder)):   
                scanID = recon_file.split('_')[-1].split('.')[0]

                # Load original k-space data
                kspace_file = os.path.join(kspace_dir, f"{scanID}_kspace.h5")
                if not os.path.exists(kspace_file):
                    logger.warning(f"Kspace file not found for patient {scanID}")
                    continue

                with h5py.File(kspace_file, "r") as f:
                    kspace_slice = f["kspace"][()]

                # Compute the center fraction for calibration
                xres = kspace_slice.shape[0]
                center_fraction = 0.08  # 8% for espirit-cf=8
                calib_size = int(round(center_fraction * xres))

                cov = compute_covariance(kspace_slice)

                for noise_increase in noise_increase_levels:
                    recon_image_path = os.path.join(recon_folder, recon_file)
                    recon_image = load_image(recon_image_path)

                    if noise_increase == 0:
                        # Use original ground truth
                        gt_image_path = os.path.join(img_dir, 'gt', f"gt_{scanID}.png")
                        gt_image = load_image(gt_image_path)
                        gt_image_orig = gt_image.clone()
                    else:
                        # Add noise based on the covariance matrix
                        kspace_noisy = add_noise_with_scaled_covariance(
                            kspace_slice, cov, noise_increase
                        )
                        kspace_noisy = kspace_noisy.transpose(2, 0, 1) 
                        im_slice, _ = process_kspace_slice(
                            kspace_noisy, 
                            calib_size=calib_size, 
                            device=device
                        )

                        gt_image = np.abs(im_slice)
                        gt_image = np.flip(gt_image)
                        gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min())

                        gt_image = torch.from_numpy(gt_image)
                        gt_image = gt_image.unsqueeze(0).unsqueeze(0) # Shape: 1x1xHxW
                        gt_image = gt_image.to(torch.float32) 

                    # Compute metrics
                    img_result = {
                        "Acceleration": acc,
                        "Method": method,
                        "ScanID": scanID,
                        "CoilNoiseIncrease": noise_increase,
                    }

                    for metric_name, metric in metrics_dict.items():
                        img_result[metric_name] = metric(recon_image.clone(), gt_image.clone()).item()

                    results.append(img_result)
                    print(img_result)

                if len(results) % 100 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(results_csv, index=False)
                    logger.info(f"Saved intermediate results with {len(results)} entries")

    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)
    logger.info(f"Saved final results with {len(results)} entries")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_dir", default="./image_dir", help="Specify the top level folder of your MR recon image directory. (default: './image_dir')")
    parser.add_argument("--kspace_dir", default="./kspace_dir", help="Specify the top level folder of your MR recon kspace directory. (default: './kspace_dir')")
    parser.add_argument("--results_dir", default="./results", help="Specify the folder to save the results csv file. (default: './results')")
    
    args = parser.parse_args()

    img_dir = args.img_dir
    kspace_dir = args.kspace_dir
    results_dir = args.results_dir

    noise_increase_levels = [0, 1, 2, 3]

    compute_metrics_acq_noise(img_dir,kspace_dir, results_dir, noise_increase_levels)