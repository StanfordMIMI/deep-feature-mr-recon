from typing import Sequence
import torch
import torch.nn as nn
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from scipy.ndimage import gaussian_laplace
import numpy as np

class HFEN(Metric):
    """
    High Frequency Error Norm (HFEN) [1]. 

    HFEN calculates the error in the high-frequency components between the predicted and target images 
    leveraging a Lapacian of Gaussians [1].

    References:
    [1] S. Ravishankar and Y. Bresler. "MR Image Reconstruction From Highly Undersampled 
        k-Space Data by Dictionary Learning," IEEE Transactions on Medical Imaging, 2011.

    Args:
        sigma (float): The standard deviation of the Gaussian used in the LoG filter.
        mode (str): Determines how to interpret the channel dimension of the inputs. One of:
                * ``'grayscale'``: Each channel corresponds to a distinct grayscale input image.
                * ``'rgb'``: The 3 channel dimensions correspond to a single rgb image.
                             Exception will be thrown if channel dimension != 3 or dtype is complex
    """
    is_differentiable = True
    higher_is_better = False

    def __init__(self, 
                sigma: float = 1.5,
                mode: str = "grayscale",
                channel_names: Sequence[str] = None,
                reduction="none",
                compute_on_step: bool = False,
                dist_sync_on_step: bool = False,
                process_group: bool = None,
                dist_sync_fn: bool = None
    ):

        super().__init__(
            channel_names=channel_names,
            units="",
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        valid_modes = ("grayscale", "rgb")
        if mode not in valid_modes:
            raise ValueError(f"Invalid `mode` ('{mode}'). Expected one of {valid_modes}.")
        
        self.mode = mode
        self.sigma = sigma

    def func(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "grayscale":
            loss_shape = (targets.shape[0], targets.shape[1])
        elif self.mode == "rgb":
            if targets.shape[1] != 3:
                raise ValueError(
                    f"Channel dimension must have size 3 for rgb mode,\
                    but got tensor of shape {targets.shape}."
                )

            is_complex = cplx.is_complex(targets) or cplx.is_complex_as_real(targets)
            if is_complex:
                raise TypeError(
                    f"Data type must be real when mode is {self.mode},\
                    but got data type {targets.dtype}"
                )

            loss_shape = (targets.shape[0], 1)

        preds = self.preprocess_hfen(preds)
        targets = self.preprocess_hfen(targets)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        LoG_preds = gaussian_laplace(preds_np, sigma = self.sigma)
        LoG_targets = gaussian_laplace(targets_np, sigma = self.sigma)

        loss = np.linalg.norm(LoG_targets - LoG_preds) / np.linalg.norm(LoG_targets)
        loss = torch.tensor(loss)
        loss = loss.view(loss_shape)

        return loss

    def preprocess_hfen(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for HFEN computation.

        Converts to a magnitude scan if complex and reshapes the tensor.

        Args:
            img (torch.Tensor): Tensor to preprocess.

        Returns:
            img (torch.Tensor): Preprocessed tensor.
        """

        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        if is_complex:
            img = cplx.abs(img)

        if self.mode == "grayscale":
            img = img.reshape(img.shape[0] * img.shape[1], 1, img.shape[2], img.shape[3])
        elif self.mode == "rgb":
            img = torch.mean(img, axis=1, keepdim=True)

        return img