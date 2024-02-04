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


def compute_metrics(img_dir, results_csv, metrics_dict):

    methods = ["UNet", "Unrolled"]
    accelerations = [2, 4, 6]
   
    results = []

    for method in methods:
        for acc in accelerations:
            folder_name = f"{method}_{acc}x"
            recon_folder = os.path.join(img_dir, folder_name)

            for recon_file in sorted(os.listdir(recon_folder)):
                scanID  = recon_file.split('_')[-1].split('.')[0]
                recon_image_path = os.path.join(recon_folder, recon_file)
                gt_image_path = os.path.join(img_dir, 'gt', f"gt_{scanID}.png")

                recon_image = load_image(recon_image_path)
                gt_image = load_image(gt_image_path)

                img_result = {
                            "Acceleration": acc,
                            "Method": method,
                            "ScanID": scanID,
                }

                for metric_name, metric in metrics_dict.items():
                    # print(metric_name, metric)
                    img_result[metric_name] = metric(recon_image, gt_image).item()
                
                results.append(img_result)
                print(img_result)

    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)

def compute_ssfd_hp_metrics(img_dir, results_dir):
    
    patch_cov_10_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                                "SSL_pretext_tasks/PatchCoverage/10/model.ckpt")
    
    patch_cov_25_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                                "SSL_pretext_tasks/PatchCoverage/25/model.ckpt")

    patch_cov_50_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                                "SSL_pretext_tasks/PatchCoverage/50/model.ckpt")

    patch_cov_75_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                                "SSL_pretext_tasks/PatchCoverage/75/model.ckpt")

    
    patch_size_1_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                           "SSL_pretext_tasks/PatchSize/1/model.ckpt")

    patch_size_4_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                           "SSL_pretext_tasks/PatchSize/4/model.ckpt")
    
    patch_size_8_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                           "SSL_pretext_tasks/PatchSize/8/model.ckpt")

    patch_size_16_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                           "SSL_pretext_tasks/PatchSize/16/model.ckpt")      

    patch_size_64_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                           "SSL_pretext_tasks/PatchSize/64/model.ckpt")                   

    metrics_dict = {"SSFD (Patch Size 1)": SSFD(ssfd_huggingface_url = patch_size_1_hf_url),
                    "SSFD (Patch Size 4)": SSFD(ssfd_huggingface_url = patch_size_4_hf_url),
                    "SSFD (Patch Size 8)": SSFD(ssfd_huggingface_url = patch_size_8_hf_url),
                    "SSFD (Patch Size 16)": SSFD(ssfd_huggingface_url = patch_size_16_hf_url),
                    "SSFD (Patch Size 64)": SSFD(ssfd_huggingface_url = patch_size_64_hf_url),

                    "SSFD (Patch Coverage 10)": SSFD(ssfd_huggingface_url = patch_cov_10_hf_url),
                    "SSFD (Patch Coverage 25)": SSFD(ssfd_huggingface_url = patch_cov_25_hf_url),
                    "SSFD (Patch Coverage 50)": SSFD(ssfd_huggingface_url = patch_cov_50_hf_url),
                    "SSFD (Patch Coverage 75)": SSFD(ssfd_huggingface_url = patch_cov_75_hf_url)
                    }
    
    results_csv = os.path.join(results_dir, "ssfd_hp_metrics.csv")

    compute_metrics(img_dir, results_csv, metrics_dict)

def compute_ssfd_percent_data_metrics(img_dir, results_dir):
    
    pct_data_5_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                         "SSL_percent_data/5/model.ckpt")
    
    pct_data_10_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                         "SSL_percent_data/10/model.ckpt")
    
    pct_data_25_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                         "SSL_percent_data/25/model.ckpt")

    pct_data_50_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                         "SSL_percent_data/50/model.ckpt")

    metrics_dict = {
                    "SSFD (5 pct data)": SSFD(ssfd_huggingface_url = pct_data_5_hf_url),
                    "SSFD (10 pct data)": SSFD(ssfd_huggingface_url = pct_data_10_hf_url),
                    "SSFD (25 pct data)": SSFD(ssfd_huggingface_url = pct_data_25_hf_url),
                    "SSFD (50 pct data)": SSFD(ssfd_huggingface_url = pct_data_50_hf_url),
    }
    
    results_csv = os.path.join(results_dir, "ssfd_pct_data_metrics.csv")

    compute_metrics(img_dir, results_csv, metrics_dict)

def compute_ssfd_fs_metrics(img_dir, results_dir):
    
    fs_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                "FatSuppression/FatSuppressed/model.ckpt")
    
    nonfs_hf_url = ("https://huggingface.co/philadamson93/SSFD/resolve/main/"
                    "FatSuppression/nonFatSuppressed/model.ckpt")

    metrics_dict = {"SSFD": SSFD(),
                    "SSFD (FS)": SSFD(ssfd_huggingface_url = fs_hf_url),
                    "SSFD (non-FS)": SSFD(ssfd_huggingface_url = nonfs_hf_url)}
    
    results_csv = os.path.join(results_dir, "ssfd_fs_metrics.csv")

    compute_metrics(img_dir, results_csv, metrics_dict)


def compute_ssfd_layer_metrics(img_dir, results_dir):

    metrics_dict = {"SSFD (layer 1)": SSFD(layer_names = ("block1_relu1",)),
                    "SSFD (layer 3)": SSFD(layer_names = ("block2_relu1",)),
                    "SSFD (layer 5)": SSFD(layer_names = ("block3_relu1",)),
                    "SSFD (layer 7)": SSFD(layer_names = ("block4_relu1",)),
                    "SSFD (layer 9)": SSFD(layer_names = ("block5_relu1",))
    }

    results_csv = os.path.join(results_dir, "layer_metrics.csv")

    compute_metrics(img_dir, results_csv, metrics_dict)


def compute_main_metrics(img_dir, results_dir):

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

    results_csv = os.path.join(results_dir, "main_metrics.csv")

    compute_metrics(img_dir, results_csv, metrics_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img_dir", default="./image_dir", help="Specify the top level folder of your MR recon image directory. (default: './image_dir')")
    parser.add_argument("--results_dir", default="./results", help="Specify the folder to save the results csv file. (default: './results')")
    
    args = parser.parse_args()

    img_dir = args.img_dir
    results_dir = args.results_dir

    compute_main_metrics(img_dir, results_dir)
    compute_SSFD_fs_metrics(img_dir, results_dir)
    compute_ssfd_hp_metrics(img_dir, results_dir)
    compute_ssfd_layer_metrics(img_dir, results_dir)
    compute_ssfd_percent_data_metrics(img_dir, results_dir)