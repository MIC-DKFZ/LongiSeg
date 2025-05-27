from longiseg.training.LongiSegTrainer.nnUNetTrainerLongi import nnUNetTrainerNoLongi
import torch


class nnUNetTrainerNoDeepSupervision(nnUNetTrainerNoLongi):
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
