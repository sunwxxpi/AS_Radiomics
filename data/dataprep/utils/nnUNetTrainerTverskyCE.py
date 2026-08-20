"""빈 예측 마스크를 줄이기 위한 recall 편향 nnU-Net trainer — Dice 항을 Tversky(beta>alpha) 로 바꾼다.

nnU-Net 은 trainer 클래스를 nnunetv2 패키지 안에서만 찾으므로 (`run_training.get_trainer_from_args`)
이 파일은 설치 스크립트가 site-packages 쪽으로 심볼릭 링크한다. 링크가 없으면 `-tr` 이름을 못 찾아 즉시 에러다.
"""

from typing import Callable

import numpy as np
import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """MemoryEfficientSoftDiceLoss 의 Tversky 판. alpha=beta=0.5 면 Dice 와 같은 값이다.

    beta 를 키우면 false negative 가 false positive 보다 비싸져 예측이 커지는 쪽으로 학습된다.
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # 이 줄부터는 no_grad 밖이어야 한다 — 안에 두면 gradient 가 끊긴다
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        false_pos = sum_pred - intersect
        false_neg = sum_gt - intersect
        tversky = (intersect + self.smooth) / torch.clip(
            intersect + self.alpha * false_pos + self.beta * false_neg + self.smooth, 1e-8)

        return -tversky.mean()


class nnUNetTrainerTverskyCE(nnUNetTrainer):
    """기본 trainer 에서 Dice 항만 Tversky 로 바꾼다. 나머지 하이퍼파라미터는 손대지 않는다."""

    tversky_alpha = 0.3
    tversky_beta = 0.7

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                'region 기반 라벨에는 Tversky 를 붙이지 않았다 — 이 데이터셋은 label 1개짜리다')

        loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp,
                               'alpha': self.tversky_alpha, 'beta': self.tversky_beta}, {},
                              weight_ce=1, weight_dice=1,
                              ignore_label=self.label_manager.ignore_label,
                              dice_class=MemoryEfficientSoftTverskyLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerTverskyCE_250epochs(nnUNetTrainerTverskyCE):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerTverskyCE_500epochs(nnUNetTrainerTverskyCE):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerTverskyCE_2000epochs(nnUNetTrainerTverskyCE):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
