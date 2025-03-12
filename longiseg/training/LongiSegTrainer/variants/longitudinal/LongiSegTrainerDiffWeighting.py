import torch

from longiseg.training.LongiSegTrainer.LongiSegTrainer import LongiSegTrainer


class LongiSegTrainerDiffWeighting(LongiSegTrainer):
    architecture_class_name = "LongiUNetDiffWeighting"


class LongiSegTrainerDiffWeightingRP(LongiSegTrainerDiffWeighting):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.random_prior = True