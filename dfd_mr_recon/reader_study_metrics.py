import os
import numpy as np
from PIL import Image
import pandas as pd

from torchvision import transforms
import torch

from metrics.ssfd import SSFD
from metrics.lpip import LPIPS
from metrics.rinfd import RINFD
from metrics.dists import DISTS
from meddlr.metrics.image import SSIM, PSNR


def load_image(image_path):
    # Load an image and convert it to a tensor
    image = Image.open(image_path).convert('L')  # Convert to grayscale if necessary
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Shape: 1x1xHxW
    return image_tensor


def compute_metrics(img_dir, results_dir, metrics_dict):

    methods = ["UNet", "Unrolled"]
    accelerations = ["2x", "4x", "6x"]
   
    results = []

    for method in methods:
        for acc in accelerations:
            folder_name = f"{method}_{acc}"
            recon_folder = os.path.join(img_dir, folder_name)

            for recon_file in sorted(os.listdir(recon_folder)):
                patient_name = recon_file.split('_')[-1].split('.')[0]
                recon_image_path = os.path.join(recon_folder, recon_file)
                gt_image_path = os.path.join(img_dir, 'gt', f"gt_{patient_name}.png")

                recon_image = load_image(recon_image_path)
                gt_image = load_image(gt_image_path)

                img_result = {
                            "Acceleration": acc,
                            "Reconstruction Method": method,
                            "Patient Name": patient_name,
                }

                for metric_name, metric in metrics_dict.items():
                    # print(metric_name, metric)
                    img_result[metric_name] = metric(recon_image, gt_image).item()
                
                results.append(img_result)
                print(img_result)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)


if __name__ == "__main__":

    img_dir = "/bmrNAS/people/padamson/results/Perceptual_Loss/Reader_Study_Data/Reader_Study_png"
    results_dir = "/bmrNAS/people/padamson/results/meddlr/ssfd/reader_study_script"

    metrics_dict = {"SSIM": SSIM(im_type = None),
                    "PSNR": PSNR(im_type = None),
                    "SSFD": SSFD(),
                    "LPIPS (VGG-16)": LPIPS(net_type='vgg', lpips=True,),
                    "LPIPS (AlexNet)": LPIPS(net_type='alex', lpips=True,),
                    "VGG-16": LPIPS(net_type='vgg', lpips=False,),
                    "DISTS": DISTS(),
                    "ResNet50 (ImageNet)": RINFD(model_weights_mode = "ImageNet"),
                    "ResNet50 (RadImageNet)": RINFD(model_weights_mode = "RadImageNet"),
                    "ResNet50 (random)": RINFD(model_weights_mode = "random")
                }

    compute_metrics(img_dir, results_dir, metrics_dict)