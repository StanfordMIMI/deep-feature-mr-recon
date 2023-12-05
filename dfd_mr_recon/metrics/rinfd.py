from typing import Sequence

import torch
from torch import nn
from torchvision.models import resnet50, ResNet, ResNet50_Weights

from meddlr.metrics.functional.image import mse
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
from meddlr.utils import env


# TODO: Refactor SSFD Class to extract shared logic into parent class FeatureMetric
class RINFD(Metric):
    """
    RadImageNet Feature Distance. RINFD evaluates the feature distance between a
    pair of images from features extracted from a pre-trained ResNet50 trained on
    the RadImageNet dataaset [1]. Other pre-training modes also available.

    References:
    ..  [1] Mei, Xueyan, et al.
        "RadImageNet: An Open Radiologic Deep Learning Research Dataset 
        or Effective Transfer Learning." Radiology: Artificial Intelligence 4.5 (2022)
        https://pubs.rsna.org/doi/full/10.1148/ryai.210315

    """

    is_differentiable = True
    higher_is_better = False

    def __init__(
        self,
        mode: str = "grayscale",
        layer_names: Sequence[str] = ("layer3.3.conv3",),
        model_weights_mode = "RadImageNet",
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
                             Exception will be thrown if channel dimension != 3 or dtype is complex
            layer_names (Sequence[str]):
                A list of strings specifying the layers to extract features from. 
                RINFD from each layer will be summed if multiple layers are specified.
            model_weights_mode (str): Which type of weights to load in the ResNet50. One of
                * ``'ImageNet'``: ResNet50 pre-trained on the ImageNet dataset.
                * ``'RadImageNet'``: ResNet50 pre-trained on the RadImageNet dataset.
                * ``'random'``: A randomly initialized, untrained ResNet50.
        """

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
        self.layer_names = layer_names
        self.model_weights_mode = model_weights_mode


        if self.model_weights_mode == 'ImageNet':
            self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        elif self.model_weights_mode == 'RadImageNet':
            path_manager = env.get_path_manager()
            RIN_weights_huggingface_url = ("https://huggingface.co/philadamson93/RINFD/resolve"
                                           "/main/ResNet50/ResNet50.pt")
            file_path = path_manager.get_local_path(RIN_weights_huggingface_url, force=False)
            self.net = resnet50(weights = torch.load(file_path))

        elif self.model_weights_mode == 'random':
            self.net = resnet50(pretrained=False)
        
        else:
            raise ValueError(
                f"Invalid `model_weights_mode` ('{model_weights_mode}'). " 
                "Expected one of 'ImageNet', 'RadImageNet', or 'random'"
            )




        self.net.eval()

    def func(self, preds, targets) -> torch.Tensor:
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

        preds = self.preprocess_rinfd(preds)
        targets = self.preprocess_rinfd(targets)


        loss = 0
        for layer in self.layer_names:
            target_features = self.get_features(self.net,layer, targets) 
            pred_features = self.get_features(self.net, layer, preds) 

            loss += torch.mean(mse(target_features, pred_features), dim=1)
        loss = loss.view(loss_shape)
        return loss

    def preprocess_rinfd(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for RINFD model input.

        Converts to a magnitude scan if complex and normalizes between [-1, 1].
        If self.mode is 'rgb', then the image will be averaged over the channel dimension.

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
            img = 2 * (img - img_min) / (img_max - img_min) - 1

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
            img = 2 * (img - img_min) / (img_max - img_min) - 1

        return img

    
    def get_features(self, model, layer_name, img: torch.Tensor):
        # Function to store the output of the target layer
        features = None

        # Define the hook function
        def hook_function(module, input, output):
            nonlocal features
            features = output

        # Register the hook
        layer = dict([*model.named_modules()])[layer_name]
        handle = layer.register_forward_hook(hook_function)

        # Forward pass
        with torch.no_grad():
            model(img)

        # Remove the hook
        handle.remove()

        return features