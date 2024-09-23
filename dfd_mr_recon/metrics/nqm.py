from typing import Sequence
import torch
import torch.nn as nn
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
import numpy as np
from scipy.fftpack import fftshift

class NQM(Metric):
    """

    Noise Quality Metric (NQM). 
    
    NQM computes the SNR of a model-restored image via a restoration algorithm that accounts for the 
    impact of spatial frequencies, distance, and contrast masking on contrast sensitivities [1]. 
    Code adapted from the open-source Image Quality Assessment library [2].
    
    References:
    [1] N. Damera-Venkata et. al. "Image quality assessment based on a degradation model." 
        IEEE transactions on image processing. 2000;9(4):636–650.

    [2] github.com/lucia15/Image-Quality-Assessment/

   
    Args:
        mode (str): Determines how to interpret the channel dimension of the inputs. One of:
                * ``'grayscale'``: Each channel corresponds to a distinct grayscale input image.
                * ``'rgb'``: The 3 channel dimensions correspond to a single rgb image.
                             Exception will be thrown if channel dimension != 3 or dtype is complex

    """
    is_differentiable = True
    higher_is_better = True

    def __init__(self, 
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
        
        self.mode = mode


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

        preds = self.preprocess_nqm(preds)
        targets = self.preprocess_nqm(targets)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        # NQM computation requires 2D inputs
        if self.mode == "grayscale":
            preds_np = np.squeeze(preds_np, axis=(0,1)) 
            targets_np = np.squeeze(targets_np, axis=(0,1))
            loss = self.nqm(preds_np, targets_np)
        elif self.mode == "rgb":
            losses = []
            for i in range(3):
                pred_channel = np.squeeze(preds_np[:, i, :, :], axis=(0, 1))
                target_channel = np.squeeze(targets_np[:, i, :, :], axis=(0, 1))
                loss_channel = self.nqm(pred_channel, target_channel)
                losses.append(loss_channel)
            loss = np.mean(losses)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        loss = torch.tensor(loss)
        loss = loss.view(loss_shape)

        return loss

    def preprocess_nqm(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for NQM computation.

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

        img *= 255 # rescale from 0-1 to 0-255

        return img


    def nqm(self, pred, target):
        """
        Compute the Noise Quality Metric (NQM) between two images.

        Args:
            pred (ndarray): Predicted image in NumPy array format.
            target (ndarray): Target/reference image in NumPy array format.

        Returns:
            float: NQM score.
        """

        (A, AI) = self.__get_correlated_images(target, pred)
        (y1, y2) = self.__reconstruct_images(A, AI)
        y = self.__compute_quality(y1, y2)
        return y

    def __ctf(self, f_r):
        """ Bandpass Contrast Threshold Function for RGB"""
        (gamma, alpha) = (0.0192 + 0.114 * f_r, (0.114 * f_r) ** 1.1)
        beta = np.exp(-alpha)
        num = 520.0 * gamma * beta
        return 1.0 / num

    def _get_masked(self, c, ci, a, ai, i):
        (H, W) = c.shape
        (c, ci, ct) = (c.flatten('F'), ci.flatten('F'), self.__ctf(i))
        ci[abs(ci) > 1.0] = 1.0
        T = ct * (0.86 * ((c / ct) - 1.0) + 0.3)
        (ai, a, a1) = (ai.flatten('F'), a.flatten('F'), (abs(ci - c) - T) < 0.0)
        ai[a1] = a[a1]
        return ai.reshape(H, W)

    def __get_thresh(self, x, T, z, trans=True):
        (H, W) = x.shape
        if trans:
            (x, z) = (x.flatten('F').T, z.flatten())
        else:
            (x, z) = (x.flatten('F'), z.flatten('F'))
        z[abs(x) < T] = 0.0
        return z.reshape(H, W)

    def __decompose_cos_log_filter(self, w1, w2, phase=np.pi):
        return 0.5 * (1 + np.cos(np.pi * np.log2(w1 + w2) - phase))

    def __get_w(self, r):
        w = [(r + 2) * ((r + 2 <= 4) * (r + 2 >= 1))]
        w += [r * ((r <= 4) * (r >= 1))]
        w += [r * ((r >= 2) * (r <= 8))]
        w += [r * ((r >= 4) * (r <= 16))]
        w += [r * ((r >= 8) * (r <= 32))]
        w += [r * ((r >= 16) * (r <= 64))]
        return w

    def __get_u(self, r):
        u = [4 * (np.logical_not((r + 2 <= 4) * (r + 2 >= 1)))]
        u += [4 * (np.logical_not((r <= 4) * (r >= 1)))]
        u += [0.5 * (np.logical_not((r >= 2) * (r <= 8)))]
        u += [4 * (np.logical_not((r >= 4) * (r <= 16)))]
        u += [0.5 * (np.logical_not((r >= 8) * (r <= 32)))]
        u += [4 * (np.logical_not((r >= 16) * (r <= 64)))]
        return u

    def __get_G(self, r):
        (w, u) = (self.__get_w(r), self.__get_u(r))
        phase = [np.pi, np.pi, 0.0, np.pi, 0.0, np.pi]
        dclf = self.__decompose_cos_log_filter
        return [dclf(w[i], u[i], phase[i]) for i in range(len(phase))]

    def __compute_fft_plane_shifted(self, ref, query):
        (x, y) = ref.shape
        (xplane, yplane) = np.mgrid[-y / 2:y / 2, -x / 2:x / 2]
        plane = (xplane + 1.0j * yplane)
        r = abs(plane)
        G = self.__get_G(r)
        Gshifted = list(map(fftshift, G))
        return [Gs.T for Gs in Gshifted]

    def __get_c(self, a, l_0):
        c = [a[0] / l_0]
        c += [a[1] / (l_0 + a[0])]
        c += [a[2] / (l_0 + a[0] + a[1])]
        c += [a[3] / (l_0 + a[0] + a[1] + a[2])]
        c += [a[4] / (l_0 + a[0] + a[1] + a[2] + a[3])]
        return c

    def __get_ci(self, ai, li_0):
        ci = [ai[0] / (li_0)]
        ci += [ai[1] / (li_0 + ai[0])]
        ci += [ai[2] / (li_0 + ai[0] + ai[1])]
        ci += [ai[3] / (li_0 + ai[0] + ai[1] + ai[2])]
        ci += [ai[4] / (li_0 + ai[0] + ai[1] + ai[2] + ai[3])]
        return ci

    def __compute_contrast_images(self, a, ai, l, li):
        ci = self.__get_ci(ai, li)
        c = self.__get_c(a, l)
        return (c, ci)

    def __get_detection_thresholds(self):
        viewing_angle = (1.0 / 3.5) * (180.0 / np.pi)
        rotations = [2.0, 4.0, 8.0, 16.0, 32.0]
        return list(map(lambda x: self.__ctf(x / viewing_angle), rotations))

    def __get_account_for_supra_threshold_effects(self, c, ci, a, ai):
        r = range(len(a))
        return [self._get_masked(c[i], ci[i], a[i], ai[i], i + 1) for i in r]

    def __apply_detection_thresholds(self, c, ci, d, a, ai):
        A = [self.__get_thresh(c[i], d[i], a[i], False) for i in range(len(a))]
        AI = [self.__get_thresh(ci[i], d[i], ai[i], True) for i in range(len(a))]
        return (A, AI)

    def __reconstruct_images(self, A, AI):
        return list(map(lambda x: np.add.reduce(x), (A, AI)))

    def __compute_quality(self, imref, imquery):
        return self.__snr(imref, imquery)

    def __get_ref_basis(self, ref_fft, query_fft, GS):
        (L_0, LI_0) = list(map(lambda x: GS[0] * x, (ref_fft, query_fft)))
        (l_0, li_0) = list(map(lambda x: np.real(np.fft.ifft2(x)), (L_0, LI_0)))
        return (l_0, li_0)

    def __compute_inverse_convolution(self, convolved_fft, GS):
        convolved = [GS[i] * convolved_fft for i in range(1, len(GS))]
        return list(map(lambda x: np.real(np.fft.ifft2(x)), convolved))

    def __correlate_in_fourier_domain(self, ref, query):
        (ref_fft, query_fft) = list(map(lambda x: np.fft.fft2(x), (ref, query)))
        GS = self.__compute_fft_plane_shifted(ref, query)
        (l_0, li_0) = self.__get_ref_basis(ref_fft, query_fft, GS)
        a = self.__compute_inverse_convolution(ref_fft, GS)
        ai = self.__compute_inverse_convolution(query_fft, GS)
        return (a, ai, l_0, li_0)

    def __get_correlated_images(self, ref, query):
        (a, ai, l_0, li_0) = self.__correlate_in_fourier_domain(ref, query)
        (c, ci) = self.__compute_contrast_images(a, ai, l_0, li_0)
        d = self.__get_detection_thresholds()
        ai = self.__get_account_for_supra_threshold_effects(c, ci, a, ai)
        return self.__apply_detection_thresholds(c, ci, d, a, ai)

    def __snr(self,reference,query):
        signal_value = (reference.astype('double') ** 2).mean()
        msev = self.__mse(reference, query)
        if msev != 0:
            value = 10.0 * np.log10(signal_value / msev)
        else:
            value = float("inf")
        return value

    def __mse(self,reference, query):
        (ref, que) = (reference.astype('double'), query.astype('double'))
        diff = ref - que
        square = (diff ** 2)
        mean = square.mean()
        return mean


