import inspect
import os
from copy import deepcopy
import shutil
from typing import Tuple, Union, List

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, save_json
from torch._dynamo import OptimizedModule

import longiseg
from longiseg.configuration import default_num_processes
from longiseg.utilities.file_path_utilities import get_output_folder
from longiseg.utilities.find_class_by_name import recursive_find_python_class
from longiseg.utilities.json_export import recursive_fix_for_json_export
from longiseg.utilities.label_handling.label_handling import determine_num_input_channels
from longiseg.utilities.plans_handling.plans_handler import PlansManager
from longiseg.utilities.utils import create_lists_from_splitted_dataset_folder_and_patients_json
from longiseg.inference.data_iterators import preprocessing_patients_iterator_fromfiles

from longiseg.inference.predict_from_raw_data import nnUNetPredictor


class LongiSegPredictor(nnUNetPredictor):
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(longiseg.__path__[0], "training", "LongiSegTrainer"),
                                                    trainer_name, 'longiseg.training.LongiSegTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in longiseg.training.LongiSegTrainer. '
                               f'Please place it there (in any .py file)!')
        print(trainer_class)
        network = trainer_class.build_network_architecture(
            trainer_class.architecture_class_name,
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def _manage_input_and_output_lists(self, source_folder: str,
                                       output_folder: Union[None, str],
                                       patient_json: dict,
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        list_of_lists = create_lists_from_splitted_dataset_folder_and_patients_json(source_folder,
                                                                                    self.dataset_json['file_ending'],
                                                                                    patient_json)
        print(f'There are {len(list_of_lists)} cases in the source folder')
        list_of_lists = list_of_lists[part_id::num_parts]
        caseids = [os.path.basename(i[0][0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in list_of_lists]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        output_filename_truncated = [join(output_folder, i) for i in caseids]

        assert folder_with_segs_from_prev_stage is None, "Cascaded is not implemented for longitudinal segmentation"
        seg_from_prev_stage_files = [None for _ in caseids]
        # remove already predicted files from the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists = [list_of_lists[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists, output_filename_truncated, seg_from_prev_stage_files

    def predict_patients(self,
                         source_folder: str,
                         output_folder: str,
                         patient_file: str,
                         save_probabilities: bool = False,
                         overwrite: bool = True,
                         num_processes_preprocessing: int = default_num_processes,
                         num_processes_segmentation_export: int = default_num_processes,
                         folder_with_segs_from_prev_stage: str = None,
                         num_parts: int = 1,
                         part_id: int = 0):
        """
        This is the default predict function for LongiSeg. It requires a source folder, output folder and patient_file.
        It will predict all cases from the patient_file which are present in the source folder.
        """
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        my_init_kwargs = {}
        for k in inspect.signature(self.predict_patients).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(
            my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
        recursive_fix_for_json_export(my_init_kwargs)
        maybe_mkdir_p(output_folder)
        save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

        # we need these two if we want to do things with the predictions like for example apply postprocessing
        save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            raise NotImplementedError("Cascaded is not implemented for longitudinal segmentation")
            # assert folder_with_segs_from_prev_stage is not None, \
            #     f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
            #     f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
            #     f' they are located via folder_with_segs_from_prev_stage'

        patient_json = load_json(patient_file)
        shutil.copy(patient_file, join(output_folder, 'patients.json'))

        # sort out input and output filenames
        list_of_lists, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(source_folder,
                                                output_folder,
                                                patient_json,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_patients_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)


def predict_longi_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with LongiSeg. This function is used when '
                                                 'you want to manually specify a folder containing a trained LongiSeg '
                                                 'model. This is useful when the LongiSeg environment variables '
                                                 '(LongiSeg_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-pat', type=str, required=True,
                        help='json file containing test patient information. Must be a dictionary with patient IDs as '
                             'keys and a list of scans as values.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X LongiSeg_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help LongiSeg if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = LongiSegPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    predictor.predict_patients(args.i, args.o, args.pat, 
                              save_probabilities=args.save_probabilities,
                              overwrite=not args.continue_prediction,
                              num_processes_preprocessing=args.npp,
                              num_processes_segmentation_export=args.nps,
                              folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                              num_parts=1, part_id=0)


def predict_longi_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with LongiSeg. This function is used when '
                                                 'you want to manually specify a folder containing a trained LongiSeg '
                                                 'model. This is useful when the LongiSeg environment variables '
                                                 '(LongiSeg_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-pat', type=str, required=True,
                        help='json file containing test patient information. Must be a dictionary with patient IDs as '
                             'keys and a list of scans as values.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='LongiSegTrainer',
                        help='What LongiSeg trainer class was used for training? Default: LongiSegTrainer')
    parser.add_argument('-c', type=str, required=False, default='3d_fullres',
                        help='LongiSeg configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p. Default: 3d_fullres')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate LongiSeg_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple LongiSeg_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 LongiSeg_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X LongiSeg_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See LongiSeg_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help LongiSeg if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = LongiSegPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.predict_patients(args.i, args.o, args.pat,
                               save_probabilities=args.save_probabilities,
                               overwrite=not args.continue_prediction,
                               num_processes_preprocessing=args.npp,
                               num_processes_segmentation_export=args.nps,
                               folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                               num_parts=args.num_parts,
                               part_id=args.part_id)