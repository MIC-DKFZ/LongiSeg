from longiseg.training.LongiSegTrainer.nnUNetTrainerLongi import nnUNetTrainerLongi
import torch


class nnUNetTrainerNoDeepSupervision(nnUNetTrainerLongi):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
