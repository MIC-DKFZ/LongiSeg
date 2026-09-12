import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import List, Sequence

import numpy as np

DEFAULT_WEIGHTS_FOLDER = os.environ.get("LongiSeg_weights", str(Path.home() / ".cache" / "longiseg"))


@contextmanager
def _working_directory(folder: str):
    # uniGradICON downloads its weights into network_weights relative to the current working directory and offers no
    # way to change that, so we run it somewhere it can do no harm instead of littering wherever the user called us
    previous = os.getcwd()
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)


class PairRegistration(ABC):
    """A registration of one baseline/follow-up pair. Implement this and PointPropagator to plug a different
    registration into LongiSeg_prepare_tracking or into an interactive tool."""

    @abstractmethod
    def propagate(self, bl_point_xyz: Sequence[float]) -> List[float]:
        """Map a point given in baseline voxel coordinates (x, y, z) to follow-up voxel coordinates."""


class PointPropagator(ABC):
    @abstractmethod
    def register(self, bl_image_file: str, fu_image_file: str) -> PairRegistration:
        """Register the two scans. Called once per scan pair, the result is reused for all of its lesions."""


class ITKPairRegistration(PairRegistration):
    def __init__(self, bl_image, fu_image, phi_fu_to_bl, phi_bl_to_fu, mapping_method: str,
                 marker_sigma_vox: float):
        self.bl_image = bl_image
        self.fu_image = fu_image
        self.phi_fu_to_bl = phi_fu_to_bl
        self.phi_bl_to_fu = phi_bl_to_fu
        self.mapping_method = mapping_method
        self.marker_sigma_vox = marker_sigma_vox

    @staticmethod
    def _voxel_to_physical(image, voxel_xyz: np.ndarray) -> np.ndarray:
        origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        direction = np.asarray(image.GetDirection(), dtype=np.float64)
        return origin + direction @ (voxel_xyz * spacing)

    @staticmethod
    def _physical_to_voxel(image, physical_xyz: np.ndarray) -> np.ndarray:
        origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        direction = np.asarray(image.GetDirection(), dtype=np.float64)
        return np.linalg.solve(direction, physical_xyz - origin) / spacing

    def _propagate_via_transform(self, bl_point_xyz: np.ndarray) -> np.ndarray:
        bl_phys = self._voxel_to_physical(self.bl_image, bl_point_xyz)
        fu_phys = np.asarray(self.phi_bl_to_fu.TransformPoint(tuple(bl_phys.tolist())), dtype=np.float64)
        return self._physical_to_voxel(self.fu_image, fu_phys)

    def _propagate_via_marker_warp(self, bl_point_xyz: np.ndarray) -> np.ndarray:
        import itk

        marker = np.zeros_like(np.asarray(self.bl_image), dtype=np.float32)
        cx, cy, cz = bl_point_xyz
        radius = max(1, int(np.ceil(3.0 * self.marker_sigma_vox)))
        x0, x1 = max(0, int(np.floor(cx)) - radius), min(marker.shape[2] - 1, int(np.ceil(cx)) + radius)
        y0, y1 = max(0, int(np.floor(cy)) - radius), min(marker.shape[1] - 1, int(np.ceil(cy)) + radius)
        z0, z1 = max(0, int(np.floor(cz)) - radius), min(marker.shape[0] - 1, int(np.ceil(cz)) + radius)

        zz, yy, xx = np.meshgrid(np.arange(z0, z1 + 1, dtype=np.float32),
                                 np.arange(y0, y1 + 1, dtype=np.float32),
                                 np.arange(x0, x1 + 1, dtype=np.float32), indexing="ij")
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        marker[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = np.exp(-d2 / (2.0 * self.marker_sigma_vox ** 2))

        marker_itk = itk.image_from_array(marker)
        marker_itk.SetOrigin(self.bl_image.GetOrigin())
        marker_itk.SetSpacing(self.bl_image.GetSpacing())
        marker_itk.SetDirection(self.bl_image.GetDirection())

        warped = itk.resample_image_filter(marker_itk, transform=self.phi_fu_to_bl,
                                           interpolator=itk.LinearInterpolateImageFunction.New(marker_itk),
                                           use_reference_image=True, reference_image=self.fu_image)
        warped_arr = np.asarray(warped)
        max_zyx = np.array(np.unravel_index(np.argmax(warped_arr), warped_arr.shape), dtype=np.float64)
        return max_zyx[::-1]

    def propagate(self, bl_point_xyz: Sequence[float]) -> List[float]:
        bl_point_xyz = np.asarray(bl_point_xyz, dtype=np.float64)
        if bl_point_xyz.shape != (3,):
            raise ValueError(f"Expected a point with 3 coordinates, got {bl_point_xyz.tolist()}.")

        size_xyz = np.asarray(self.bl_image.GetLargestPossibleRegion().GetSize(), dtype=np.float64) - 1.0
        if np.any(bl_point_xyz < 0.0) or np.any(bl_point_xyz > size_xyz):
            raise ValueError(f"Baseline point {bl_point_xyz.tolist()} lies outside the image, which covers "
                             f"[0, {size_xyz.tolist()}].")

        if self.mapping_method == "transform":
            fu_point_xyz = self._propagate_via_transform(bl_point_xyz)
        elif self.mapping_method == "marker_warp":
            fu_point_xyz = self._propagate_via_marker_warp(bl_point_xyz)
        else:
            raise ValueError(f"Unknown mapping_method {self.mapping_method}, expected 'transform' or 'marker_warp'.")
        return [float(c) for c in fu_point_xyz]


class UniGradIconPropagator(PointPropagator):
    def __init__(self, model: str = "unigradicon", io_sim: str = "lncc", io_iterations: int = 50,
                 mapping_method: str = "transform", marker_sigma_vox: float = 1.5, modality: str = "ct",
                 weights_folder: str = DEFAULT_WEIGHTS_FOLDER):
        self.weights_folder = weights_folder
        self.model = model
        self.io_sim = io_sim
        # uniGradICON expects None rather than 0 to mean no instance optimization
        self.io_iterations = io_iterations or None
        self.mapping_method = mapping_method
        self.marker_sigma_vox = marker_sigma_vox
        self.modality = modality
        self._net = None

    def _build_net(self):
        try:
            from unigradicon import get_model_from_model_zoo, make_sim
        except ImportError as e:
            raise ImportError(
                "Propagating points requires uniGradICON, which is an optional dependency of LongiSeg. Install it by "
                "running `pip install -e .[tracking]` in your LongiSeg clone, or provide fu_point_prop yourself."
            ) from e
        with _working_directory(self.weights_folder):
            return get_model_from_model_zoo(model_name=self.model, loss_fn=make_sim(self.io_sim))

    def register(self, bl_image_file: str, fu_image_file: str) -> ITKPairRegistration:
        import itk
        import icon_registration.itk_wrapper
        from unigradicon import preprocess

        if self._net is None:
            self._net = self._build_net()

        bl_image = itk.imread(str(bl_image_file))
        fu_image = itk.imread(str(fu_image_file))
        phi_fu_to_bl, phi_bl_to_fu = icon_registration.itk_wrapper.register_pair(
            self._net, preprocess(bl_image, self.modality), preprocess(fu_image, self.modality),
            finetune_steps=self.io_iterations)
        return ITKPairRegistration(bl_image, fu_image, phi_fu_to_bl, phi_bl_to_fu, self.mapping_method,
                                   self.marker_sigma_vox)
