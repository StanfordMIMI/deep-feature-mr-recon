"""Test the ssfd loss with motion."""
import os

import pandas as pd
from typing import Iterable, Union

import torch

from meddlr.config import get_cfg
from meddlr.data.build import build_recon_val_loader
from meddlr.data.transforms.transform import build_normalizer
from meddlr.engine import default_argument_parser, default_setup
from meddlr.evaluation import inference_on_dataset
from meddlr.evaluation.recon_evaluation import ReconEvaluator
from meddlr.forward.mri import SenseModel
from meddlr.ops import complex as cplx
from meddlr.transforms.builtin.mri import MRIReconAugmentor
from meddlr.transforms.transform import NoOpTransform, Transform, TransformList
from meddlr.transforms.transform_gen import TransformGen
from meddlr.utils.general import move_to_device
from meddlr.utils.logger import setup_logger

_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
logger = None  # initialize in setup()


class PerturbationTransform:
    """A data transform that perturbs the k-space data."""

    def __init__(self, cfg):
        """
        Args:
            motion_gen: The motion transform generator.
        """
        self.cfg = cfg
        self._normalizer = build_normalizer(cfg)

    def _generate_seed(self, fname: str, slice_id: int):
        return sum(ord(x) for x in fname + str(slice_id)) % (2**32 - 1)

    def __call__(
        self,
        kspace: torch.Tensor,
        maps: torch.Tensor,
        target: torch.Tensor,
        fname: str,
        slice_id: int,
        is_unsupervised: bool = False,
        fixed_acc: bool = False,
    ):
        kspace = torch.as_tensor(kspace).unsqueeze(0)
        maps = torch.as_tensor(maps).unsqueeze(0)
        target = torch.as_tensor(target).unsqueeze(0)

        # Seed the generator so that we have random, but reproducible, motion artifacts.
        # augmentor = copy.deepcopy(self.augmentor)
        seed = self._generate_seed(fname, slice_id)
        augmentor = MRIReconAugmentorPerturbation.from_cfg(cfg, aug_kind="aug_train", device="cpu", seed=seed)

        # Corrupt the kspace.
        out, _, _ = augmentor(kspace=kspace, maps=maps, target=target, normalizer=self._normalizer)
        if out["mean"] is None:
            out["mean"] = torch.as_tensor([0.0])
        if out["std"] is None:
            out["std"] = torch.as_tensor([1.0])

        # Coil-combined image.
        # FIXME: This may be faster to do outside of the data transform.
        # If data loading is slow, try moving this outside.
        # NOTE: If you do move it outside and comptue on the GPU, there is no guarantee the results
        # will be the same as on the cpu. Keep this consistent in your analysis.
        out["pred"] = SenseModel(maps=out["maps"])(out["kspace"], adjoint=True)

        # Remove batch dimension.
        out.update({k: out[k].squeeze(0) for k in ["kspace", "pred", "target", "maps"]})
        return out


class NoOpModel(torch.nn.Module):
    def forward(self, inputs):
        inputs = move_to_device(inputs, cfg.MODEL.DEVICE)
        return {"pred": inputs.pop("pred"), "target": inputs.pop("target")}

class MRIReconAugmentorPerturbation(MRIReconAugmentor):
    '''Modification of MRIReconAugmentor to only apply transforms to input, not target.'''

    def _apply_te(
        self,
        tfms_equivariant: Iterable[Union[Transform, TransformGen]],
        image: torch.Tensor,
        target: torch.Tensor,
        maps: torch.Tensor,
    ):
        """Apply equivariant transforms for perturbation analysis.

        These transforms typically affect both the input and the target, but here
        we modify such that it only affects the input.

        Args:
            tfms_equivariant: Equivariant transforms to apply.
            image: The kspace to apply these transformations to.
            target:

        Returns:
            Tuple[torch.Tensor, TransformList]: The transformed kspace and the list
                of deterministic transformations that were applied.
        """
        tfms = []
        for g in tfms_equivariant:
            tfm: Transform = g.get_transform(image) if isinstance(g, TransformGen) else g
            if isinstance(tfm, NoOpTransform):
                continue
            image = tfm.apply_image(image)
            tfms.append(tfm)
        return image, target, maps, TransformList(tfms, ignore_no_op=True)


def build_data_loader(cfg, dataset_name: str, transform: PerturbationTransform):
    dl = build_recon_val_loader(
        cfg=cfg, dataset_name=dataset_name, as_test=True, add_noise=False, add_motion=False
    )
    dl.dataset.transform = transform
    return dl


def setup(args):
    """
    Create configs and perform basic setups.
    We do not save the config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if opts and opts[0] == "--":
        opts = opts[1:]
    cfg.merge_from_list(opts)
    cfg.freeze()
    default_setup(cfg, args, save_cfg=False)

    # Setup logger for test results
    global logger
    logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, args.save_dir), name=_FILE_NAME)

    return cfg


def default_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--save-dir", type=str, default="test_results", help="Directory to save test results."
    )
    return parser


def run_eval(cfg, dataset: str):
    model = NoOpModel()

    logger.info("==" * 40)
    logger.info("Evaluating {} ...".format(dataset))
    logger.info("==" * 40)

    save_dir = os.path.join(cfg.OUTPUT_DIR, args.save_dir, dataset)
    evaluator = ReconEvaluator(
        dataset_name=dataset,
        cfg=cfg,
        device=cfg.MODEL.DEVICE,
        output_dir=save_dir,
        save_scans=True,
        metrics=cfg.TEST.VAL_METRICS.RECON,
        # layer_names=cfg.TEST.VAL_METRICS.LAYER_NAMES,
        flush_period=cfg.TEST.FLUSH_PERIOD,
        prefix="test",
        group_by_scan=True,
    )

    transform = PerturbationTransform(cfg)
    dl = build_data_loader(cfg, dataset, transform)

    results = inference_on_dataset(model, data_loader=dl, evaluator=evaluator)
    results = pd.DataFrame(results).T.reset_index().rename(columns={"index": "scan_name"})
    return results


if __name__ == "__main__":
    args = default_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    datasets = cfg.DATASETS.TEST

    results = []
    for dataset in datasets:
        results.append(run_eval(cfg.clone(), dataset))
    results = pd.concat(results)
    results.to_csv(os.path.join(cfg.OUTPUT_DIR, args.save_dir, "test_results.csv"), index=False)
