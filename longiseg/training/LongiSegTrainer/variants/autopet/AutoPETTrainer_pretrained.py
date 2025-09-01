import os
import torch
from longiseg.training.LongiSegTrainer.variants.autopet.AutoPETTrainer import AutoPETTrainer, AutoPETTrainerCrossSecMask


class AutoPETTrainerPretrained1e3(AutoPETTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3


class AutoPETTrainerCrossSecMaskPretrained1e3(AutoPETTrainerCrossSecMask):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3