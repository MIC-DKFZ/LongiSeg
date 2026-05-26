import torch

from longiseg.training.LongiSegTrainer.variants.tracking.LongiSegTrainerTrackingDiffWeighting import LongiSegTrainerTrackingDiffWeighting


class LongiSegTrainerTrackingDiffWeightingFinetuning(LongiSegTrainerTrackingDiffWeighting):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3