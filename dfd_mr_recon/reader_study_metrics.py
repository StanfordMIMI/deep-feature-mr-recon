import os
import argparse
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
    image = Image.open(image_path).convert('L') # Convert to grayscale
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # Shape: 1x1xHxW
    return image_tensor


def compute_metrics(img_dir, results_dir, metrics_dict):

    methods = ["UNet", "Unrolled"]
    accelerations = [2.0, 4.0, 6.0]
   
    results = []

    for method in methods:
        for acc in accelerations:
            folder_name = f"{method}_{acc}"
            recon_folder = os.path.join(img_dir, folder_name)

            for recon_file in sorted(os.listdir(recon_folder)):
                scanID = recon_file.split('_')[-1].split('.')[0]
                recon_image_path = os.path.join(recon_folder, recon_file)
                gt_image_path = os.path.join(img_dir, 'gt', f"gt_{patient_name}.png")

                recon_image = load_image(recon_image_path)
                gt_image = load_image(gt_image_path)

                img_result = {
                            "Acceleration": acc,
                            "Method": method,
                            "ScanID": scanID,
                }

                for metric_name, metric in metrics_dict.items():
                    img_result[metric_name] = metric(recon_image, gt_image).item()
                
                results.append(img_result)
                print(img_result)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_dir", default="./image_dir", help="Specify the top level folder of your MR recon image directory. (default: './image_dir')")
    parser.add_argument("--results_dir", default="./results", help="Specify the folder to save the results csv file. (default: './results')")
    
    args = parser.parse_args()

    img_dir = args.img_dir
    results_dir = args.results_dir

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