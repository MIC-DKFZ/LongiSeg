from longiseg.training.LongiSegTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainer_onlyMirror01
from longiseg.training.LongiSegTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5

class nnUNetTrainer_onlyMirror01_DA5(nnUNetTrainer_onlyMirror01, nnUNetTrainerDA5):
    pass
