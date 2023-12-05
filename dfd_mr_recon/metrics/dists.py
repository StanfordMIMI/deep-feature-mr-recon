from typing import Sequence

import torch

from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from meddlr.utils import env

if env.package_available("DISTS_pytorch"):
    from DISTS_pytorch import DISTS as _DISTS


# TODO: Refactor SSFD Class to extract shared logic into parent class FeatureMetric
class DISTS(Metric):
    """
    Deep Image Structure and Texture Similarity

    DISTS is a full-reference image quality metric using vgg-16 features, with
    additional constraints to give explicit tolerance to structure resampling [1].
    DISTS has been shown to correspond well to perceived image quality on
    natural images [1] and MR images [2].

    References:
    ..  [1] Ding, Keyan, et al. "Image quality assessment: Unifying structure and
        texture similarity." IEEE transactions on pattern analysis and machine
        intelligence 44.5 (2020): 2567-2581

        [2] Kastryulin, Sergey, et al. "Image quality assessment for
            magnetic resonance imaging." IEEE Access 11 (2023): 14154-14168.
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        mode: str = "grayscale",
        channel_names: Sequence[str] = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
    ):
        """
        Args:
        mode (str): Determines how to interpret the channel dimension of the inputs. One of:
            * ``'grayscale'``: Each channel corresponds to a distinct grayscale input image.
            * ``'rgb'``: The 3 channel dimensions correspond to a single rgb image.
                         Exception will be thrown if channel dimension != 3 dtype data is complex.
        """

        if not env.package_available("DISTS_pytorch"):
            raise ModuleNotFoundError(
                "DISTS metric requires that dists-pytorch is installed."
                "Either install as `pip install meddlr[metrics]` or `pip install dists-pytorch`."
            )

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

        self.net = _DISTS()
        self.mode = mode

    def func(self, preds: torch.Tensor, targets: torch.Tensor):

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

        preds = self.preprocess_dists(preds)
        targets = self.preprocess_dists(targets)

        loss = self.net(preds, targets)
        loss = loss.view(loss_shape)

        return loss

    def preprocess_dists(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image per DISTS implementation.

        Converts images to magnitude images if complex and normalizes between [0, 1].
        If self.mode is 'grayscale', then each channel dimension will be replicated 3 times.

        Args:
            img (torch.Tensor): Tensor to preprocess.

        Returns:
            img (torch.Tensor): Preprocessed tensor.
        """

        is_complex = cplx.is_complex(img) or cplx.is_complex_as_real(img)
        if is_complex:
            img = cplx.abs(img)

        if self.mode == "grayscale":
            # normalize each image independently (channel dim. represents different images)
            shape = (img.shape[0], img.shape[1], -1)
            img_min = torch.amin(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1)
            img_max = torch.amax(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1)
            img = (img - img_min) / (img_max - img_min)

            img = img.reshape(img.shape[0] * img.shape[1], 1, img.shape[2], img.shape[3])
            img = img.repeat(1, 3, 1, 1)
        elif self.mode == "rgb":
            # normalize each image independently (channel dim. represents the same image)
            shape = (img.shape[0], -1)
            img_min = (
                torch.amin(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            )
            img_max = (
                torch.amax(img.reshape(shape), dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            )
            img = (img - img_min) / (img_max - img_min)

        return img
