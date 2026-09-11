from longiseg.training.LongiSegTrainer.variants.tracking.LongiSegTrainerTrackingFinetuning import \
    LongiSegTrainerTrackingFinetuning


class LongiSegTrainerTrackingDiffWeightingFinetuning(LongiSegTrainerTrackingFinetuning):
    architecture_class_name = "LongiUNetTrackingDiffWeighting"
