from typing import Sequence
import torch
import torch.nn as nn
from meddlr.metrics.metric import Metric
from meddlr.ops import complex as cplx
import numpy as np

class VIF(Metric):
    """

    Visual Information Fidelity (VIF) uses natural image scene statistics to compute the mutual information between image pairs [1]. 
    This implementation uses the steerable pyramid version of VIF proposed in [1] and is based on the github repository [2].

    [1]H.R. Sheikh, A.C. Bovik and G. de Veciana, "An information fidelity criterion for image quality assessment using natural scene statistics," 
    IEEE Transactions on Image Processing , vol.14, no.12pp. 2117- 2128, Dec. 2005.

    [2] https://github.com/abhinaukumar/vif

  
    Args:

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
        """
            Args:
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

        preds = self.preprocess_vif(preds)
        targets = self.preprocess_vif(targets)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        if self.mode == "grayscale":
            preds_np = np.squeeze(preds_np, axis=(0,1)) 
            targets_np = np.squeeze(targets_np, axis=(0,1))
            loss = self.vif(preds_np, targets_np, wavelet='steerable')
        elif self.mode == "rgb":
            losses = []
            for i in range(3):
                pred_channel = np.squeeze(preds_np[:, i, :, :], axis=(0, 1))
                target_channel = np.squeeze(targets_np[:, i, :, :], axis=(0, 1))
                loss_channel = self.vif(pred_channel, target_channel, wavelet='steerable')
                losses.append(loss_channel)
            loss = np.mean(losses)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        loss = torch.tensor(loss)
        loss = loss.view(loss_shape)

        return loss

    def preprocess_vif(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for VIF computation.

        Converts to a magnitude scan if complex and reshapes the tensor.
        Rescale to 0-255.

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



    def im2col(self, img, k, stride=1):
        m, n = img.shape
        s0, s1 = img.strides
        nrows = m - k + 1
        ncols = n - k + 1
        shape = (k, k, nrows, ncols)
        arr_stride = (s0, s1, s0, s1)

        ret = np.lib.stride_tricks.as_strided(img, shape=shape, strides=arr_stride)
        return ret[:, :, ::stride, ::stride].reshape(k*k, -1)


    def integral_image(self, x):
        M, N = x.shape
        int_x = np.zeros((M+1, N+1))
        int_x[1:, 1:] = np.cumsum(np.cumsum(x, 0), 1)
        return int_x


    def moments(self, x, y, k, stride):
        kh = kw = k

        k_norm = k**2

        x_pad = np.pad(x, int((kh - stride)/2), mode='reflect')
        y_pad = np.pad(y, int((kw - stride)/2), mode='reflect')

        int_1_x = self.integral_image(x_pad)
        int_1_y = self.integral_image(y_pad)

        int_2_x = self.integral_image(x_pad*x_pad)
        int_2_y = self.integral_image(y_pad*y_pad)

        int_xy = self.integral_image(x_pad*y_pad)

        mu_x = (int_1_x[:-kh:stride, :-kw:stride] - int_1_x[:-kh:stride, kw::stride] - int_1_x[kh::stride, :-kw:stride] + int_1_x[kh::stride, kw::stride])/k_norm
        mu_y = (int_1_y[:-kh:stride, :-kw:stride] - int_1_y[:-kh:stride, kw::stride] - int_1_y[kh::stride, :-kw:stride] + int_1_y[kh::stride, kw::stride])/k_norm

        var_x = (int_2_x[:-kh:stride, :-kw:stride] - int_2_x[:-kh:stride, kw::stride] - int_2_x[kh::stride, :-kw:stride] + int_2_x[kh::stride, kw::stride])/k_norm - mu_x**2
        var_y = (int_2_y[:-kh:stride, :-kw:stride] - int_2_y[:-kh:stride, kw::stride] - int_2_y[kh::stride, :-kw:stride] + int_2_y[kh::stride, kw::stride])/k_norm - mu_y**2

        cov_xy = (int_xy[:-kh:stride, :-kw:stride] - int_xy[:-kh:stride, kw::stride] - int_xy[kh::stride, :-kw:stride] + int_xy[kh::stride, kw::stride])/k_norm - mu_x*mu_y

        mask_x = (var_x < 0)
        mask_y = (var_y < 0)

        var_x[mask_x] = 0
        var_y[mask_y] = 0

        cov_xy[mask_x + mask_y] = 0

        return (mu_x, mu_y, var_x, var_y, cov_xy)


    def vif_gsm_model(self, pyr, subband_keys, M):
        tol = 1e-15
        s_all = []
        lamda_all = []

        for subband_key in subband_keys:
            y = pyr[subband_key]
            y_size = (int(y.shape[0]/M)*M, int(y.shape[1]/M)*M)
            y = y[:y_size[0], :y_size[1]]

            y_vecs = self.im2col(y, M, 1)
            cov = np.cov(y_vecs)
            lamda, V = np.linalg.eigh(cov)
            lamda[lamda < tol] = tol
            cov = V@np.diag(lamda)@V.T

            y_vecs = self.im2col(y, M, M)

            s = np.linalg.inv(cov)@y_vecs
            s = np.sum(s * y_vecs, 0)/(M*M)
            s = s.reshape((int(y_size[0]/M), int(y_size[1]/M)))

            s_all.append(s)
            lamda_all.append(lamda)

        return s_all, lamda_all


    def vif_channel_est(self, pyr_ref, pyr_dist, subband_keys, M):
        tol = 1e-15
        g_all = []
        sigma_vsq_all = []

        for i, subband_key in enumerate(subband_keys):
            y_ref = pyr_ref[subband_key]
            y_dist = pyr_dist[subband_key]

            lev = int(np.ceil((i+1)/2))
            winsize = 2**lev + 1

            y_size = (int(y_ref.shape[0]/M)*M, int(y_ref.shape[1]/M)*M)
            y_ref = y_ref[:y_size[0], :y_size[1]]
            y_dist = y_dist[:y_size[0], :y_size[1]]

            mu_x, mu_y, var_x, var_y, cov_xy = self.moments(y_ref, y_dist, winsize, M)

            g = cov_xy / (var_x + tol)
            sigma_vsq = var_y - g*cov_xy

            g[var_x < tol] = 0
            sigma_vsq[var_x < tol] = var_y[var_x < tol]
            var_x[var_x < tol] = 0

            g[var_y < tol] = 0
            sigma_vsq[var_y < tol] = 0

            sigma_vsq[g < 0] = var_y[g < 0]
            g[g < 0] = 0

            sigma_vsq[sigma_vsq < tol] = tol

            g_all.append(g)
            sigma_vsq_all.append(sigma_vsq)

        return g_all, sigma_vsq_all


    def vif(self, img_ref, img_dist, wavelet='steerable', full=False):
        assert wavelet in ['steerable', 'haar', 'db2', 'bio2.2'], 'Invalid choice of wavelet'
        M = 3
        sigma_nsq = 0.1

        if wavelet == 'steerable':
            from pyrtools.pyramids import SteerablePyramidSpace as SPyr
            pyr_ref = SPyr(img_ref, 4, 5, 'reflect1').pyr_coeffs
            pyr_dist = SPyr(img_dist, 4, 5, 'reflect1').pyr_coeffs
            subband_keys = []
            for key in list(pyr_ref.keys())[1:-2:3]:
                subband_keys.append(key)
        else:
            from pywt import wavedec2
            ret_ref = wavedec2(img_ref, wavelet, 'reflect', 4)
            ret_dist = wavedec2(img_dist, wavelet, 'reflect', 4)
            pyr_ref = {}
            pyr_dist = {}
            subband_keys = []
            for i in range(4):
                pyr_ref[(3-i, 0)] = ret_ref[i+1][0]
                pyr_ref[(3-i, 1)] = ret_ref[i+1][1]
                pyr_dist[(3-i, 0)] = ret_dist[i+1][0]
                pyr_dist[(3-i, 1)] = ret_dist[i+1][1]
                subband_keys.append((3-i, 0))
                subband_keys.append((3-i, 1))
            pyr_ref[4] = ret_ref[0]
            pyr_dist[4] = ret_dist[0]

        subband_keys.reverse()
        n_subbands = len(subband_keys)

        [g_all, sigma_vsq_all] = self.vif_channel_est(pyr_ref, pyr_dist, subband_keys, M)

        [s_all, lamda_all] = self.vif_gsm_model(pyr_ref, subband_keys, M)

        nums = np.zeros((n_subbands,))
        dens = np.zeros((n_subbands,))
        for i in range(n_subbands):
            g = g_all[i]
            sigma_vsq = sigma_vsq_all[i]
            s = s_all[i]
            lamda = lamda_all[i]

            n_eigs = len(lamda)

            lev = int(np.ceil((i+1)/2))
            winsize = 2**lev + 1
            offset = (winsize - 1)/2
            offset = int(np.ceil(offset/M))

            g = g[offset:-offset, offset:-offset]
            sigma_vsq = sigma_vsq[offset:-offset, offset:-offset]
            s = s[offset:-offset, offset:-offset]

            for j in range(n_eigs):
                nums[i] += np.mean(np.log(1 + g*g*s*lamda[j]/(sigma_vsq+sigma_nsq)))
                dens[i] += np.mean(np.log(1 + s*lamda[j]/sigma_nsq))

        if not full:
            return np.mean(nums + 1e-4)/np.mean(dens + 1e-4)
        else:
            return np.mean(nums + 1e-4)/np.mean(dens + 1e-4), (nums + 1e-4), (dens + 1e-4)