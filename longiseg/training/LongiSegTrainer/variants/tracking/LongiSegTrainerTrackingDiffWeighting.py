from longiseg.training.LongiSegTrainer.variants.tracking.LongiSegTrainerTracking import LongiSegTrainerTracking, \
    LongiSegTrainerTrackingPretrain


class LongiSegTrainerTrackingDiffWeighting(LongiSegTrainerTracking):
    architecture_class_name = "LongiUNetTrackingDiffWeighting"


class LongiSegTrainerTrackingDiffWeightingPretrain(LongiSegTrainerTrackingPretrain):
    architecture_class_name = "LongiUNetTrackingDiffWeighting"
