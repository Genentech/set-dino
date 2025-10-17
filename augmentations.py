import logging
from enum import Enum
from functools import reduce

import cv2
import numpy as np
from typing import Optional, Tuple


class Augmentation:

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def augment(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) == 3
        augmented = self._augment_impl(img)
        return augmented

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, img: np.ndarray):
        return self.augment(img)

    @staticmethod
    def all() -> tuple['Augmentation', ...]:
        return Identity.all() + Flip.all() + Rotate.all() + ScaleAndOffset.all() + Zoom.all()

    @property
    def identifier(self) -> str:
        raise NotImplementedError()


class Identity(Augmentation):

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        return img

    @property
    def identifier(self) -> str:
        return 'identity'

    @staticmethod
    def all() -> tuple['Identity']:
        return Identity(),


class Flip(Augmentation):
    class Direction(Enum):
        UP_DOWN = 1
        LEFT_RIGHT = 2

    def __init__(self, direction: Direction):
        super().__init__()
        self._direction = direction

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        return np.flip(img, axis=self._direction.value)

    @property
    def identifier(self) -> str:
        return f'flip-{self._direction.name.lower()}'

    @staticmethod
    def up_down() -> 'Flip':
        return Flip(Flip.Direction.UP_DOWN)

    @staticmethod
    def left_right() -> 'Flip':
        return Flip(Flip.Direction.LEFT_RIGHT)

    @staticmethod
    def all() -> tuple['Flip', 'Flip']:
        return Flip.up_down(), Flip.left_right()


class Rotate(Augmentation):
    class Step(Enum):
        QUARTER = 1
        HALF = 2
        THREE_QUARTER = 3

    def __init__(self, step: Step):
        super().__init__()
        self._step = step

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        return np.rot90(img, k=self._step.value, axes=(1, 2))

    @property
    def identifier(self) -> str:
        return f'rotate-{self._step.name.lower().replace("_", "-")}'

    @staticmethod
    def quarter() -> 'Rotate':
        return Rotate(Rotate.Step.QUARTER)

    @staticmethod
    def half() -> 'Rotate':
        return Rotate(Rotate.Step.HALF)

    @staticmethod
    def three_quarter() -> 'Rotate':
        return Rotate(Rotate.Step.THREE_QUARTER)

    @staticmethod
    def all() -> tuple['Rotate', ...]:
        return Rotate.quarter(), Rotate.half(), Rotate.three_quarter()


class ScaleAndOffset(Augmentation):

    def __init__(self, scale_range: float, offset_range: float, rng: np.random.Generator = None, ):
        super().__init__()
        self._rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self._scale_mean = 1.0
        self._scale_range = scale_range # kaggle winner uses 0.1
        self._offset_mean = 0.0
        self._offset_range = offset_range  # kaggle winner uses 0.1

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        size = (img.shape[0], 1, 1)
        scale = self._rng.uniform(low=self._scale_mean-self._scale_range,
                                  high=self._scale_mean+self._scale_range,
                                  size=size)
        offset = self._rng.uniform(low=self._offset_mean-self._offset_range,
                                   high=self._offset_mean+self._offset_range,
                                   size=size)
        self._logger.debug(f'{scale=}')
        self._logger.debug(f'{offset=}')
        img = img ** scale + offset
        img = np.clip(img, 0, 1)
        return img

    @property
    def identifier(self):
        return 'scale-and-offset'

    @staticmethod
    def default_rng():
        return ScaleAndOffset()

    @staticmethod
    def all() -> tuple['ScaleAndOffset', ...]:
        return ScaleAndOffset(),


class Zoom(Augmentation):

    def __init__(self, rng: np.random.Generator = None, scale: float = 0.5):
        super().__init__()
        self._rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self._scale = scale

        assert 0 < self._scale <= 1.0

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        if self._scale == 1.0:
            return img

        scale = self._rng.uniform(low=self._scale, high=1.0)
        self._logger.debug(f'{scale=}')
        size = tuple(round(d * scale) for d in img.shape[1:])
        max_offset = np.array(img.shape[1:]) - np.array(size)
        offset = self._rng.integers(0, max_offset + 1)
        slices = (slice(None),) + tuple(slice(o, o + s) for o, s in zip(offset, size))
        self._logger.debug(f'{size=}')
        self._logger.debug(f'{offset=}')
        self._logger.debug(f'{img.shape=}')
        self._logger.debug(f'{slices=}')
        crop = img[slices]
        return np.asarray([self._resize_slice(s, img.shape[1:]) for s in crop])

    def _resize_slice(self, cropped_slice, target_dim):
        self._logger.debug(f'{cropped_slice.shape=}')
        self._logger.debug(f'{target_dim=}')
        return cv2.resize(cropped_slice, target_dim, interpolation=cv2.INTER_NEAREST)

    @property
    def identifier(self) -> str:
        return 'zoom'

    @staticmethod
    def all() -> tuple['Zoom', ...]:
        return Zoom(),


class GenerateLocalCrops(Augmentation):
    """
    Local cropping from the central regions. If the local crop is taken from the boundary, it may not include any
    cellular region and impact the model optimization.
    This class ensures local crops contain cell regions.
    """
    def __init__(self, rng: np.random.Generator = None, scale_low: float = 0.4, scale_upper: float = 0.6,
                 resize: int = 48, offset: int = 20):
        super().__init__()
        self._rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self._scale_low = scale_low
        self._scale_upper = scale_upper
        self._resize = resize
        self._offset = offset

        assert 0 < self._scale_low <= 1.0
        assert 0 < self._scale_upper <= 1.0
        assert self._scale_low <= self._scale_upper

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        if self._scale_low == 1.0:
            return img

        # sample a center point
        h, w = img.shape[1:]
        center_h = self._rng.integers(max(0, h // 2 - self._offset), min(h, h // 2 + self._offset), 1)[0]
        center_w = self._rng.integers(max(0, w // 2 - self._offset), min(w, w // 2 + self._offset), 1)[0]
        scale_h = self._rng.uniform(low=self._scale_low, high=self._scale_upper)
        scale_w = self._rng.uniform(low=self._scale_low, high=self._scale_upper)

        self._logger.debug(f'{scale_h=}')
        self._logger.debug(f'{scale_w=}')
        self._logger.debug(f'{center_h=}')
        self._logger.debug(f'{center_w=}')

        offset_h = int(scale_h * h) // 2
        offset_w = int(scale_w * w) // 2
        crop = img[:, max(0, center_h - offset_h):min(h, center_h + offset_h),
               max(0, center_w - offset_w):min(w, center_w + offset_w)]

        resized_crop = np.asarray([self._resize_slice(s, [self._resize, self._resize]) for s in crop])
        return resized_crop

    def _resize_slice(self, cropped_slice, target_dim):
        self._logger.debug(f'{cropped_slice.shape=}')
        self._logger.debug(f'{target_dim=}')
        return cv2.resize(cropped_slice, target_dim, interpolation=cv2.INTER_NEAREST)

    @property
    def identifier(self) -> str:
        return 'local_crop'


class ApplyAugmentationRandomly(Augmentation):

    def __init__(self, augmentation: Augmentation, probability: float = 0.5, rng: np.random.Generator = None):
        super().__init__()
        self._augmentation = augmentation
        self._probability = probability
        self._rng = np.random.default_rng() if rng is None else rng
        self._augmented_count = 0
        self._identity_count = 0
        self._total_count = 0

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        self._total_count += 1

        if self._rng.random() < self._probability:
            self._augmented_count += 1
            return self._augmentation.augment(img)

        self._identity_count += 1
        return img

    @property
    def identifier(self) -> str:
        return f'apply-augmentation-randomly-p={self._probability}-{self._augmentation.identifier}'


class ApplyListOfAugmentations(Augmentation):

    def __init__(self, *augmentations):
        super().__init__()
        self._augmentations = augmentations

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        return reduce(lambda i, a: a.augment(i), self._augmentations, img)

    @property
    def identifier(self) -> str:
        return f'apply-list-of-augmentations-{self._augmentations}'


class AsContiguousArray(Augmentation):

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(img)

    @property
    def identifier(self) -> str:
        return 'ascontiguousarray'


def zscore(img):
    mean = np.mean(img, axis=(1, 2))
    std = np.std(img, axis=(1, 2)) +  1e-6
    img_norm = (img - mean[..., np.newaxis, np.newaxis]) / std[..., np.newaxis, np.newaxis]
    return img_norm


class Normalization():
    """
    Channel-wise image standardization. From our analysis, 'ntc' yielded the best performance, with 'zscore' coming in
    the second place.
    """
    def __init__(self, method: str, ntc_stats_file: Optional[str] = None):
        super().__init__()
        self.method = method
        if self.method == 'ntc':
            assert ntc_stats_file is not None
            self.ntc_stats = np.load(ntc_stats_file, allow_pickle=True).item()
        print(f"====== {method} =====")

    def __call__(self, img: np.ndarray, well: Optional[Tuple[str, str]] = None) -> np.ndarray:
        """
        :param img: in shape of (channel, height, width)
        :return:
        """
        if self.method == 'clip':
            high = 99.9
            low = 0.1
            value_max = np.percentile(img, high, axis=(1, 2))
            value_min = np.percentile(img, low, axis=(1, 2))
            img_clipped = np.clip(img, value_min[:, np.newaxis, np.newaxis], value_max[:, np.newaxis, np.newaxis])
            delta_value = value_max - value_min + 1e-6
            img_norm = (img_clipped - value_min[:, np.newaxis, np.newaxis]) / delta_value[..., np.newaxis, np.newaxis]

        elif self.method == 'log':
            img[img < 1] = 1
            img = np.log(img)
            img_norm = zscore(img)

        elif self.method == 'arcsinh':
            img = np.arcsinh(img)
            img_norm = zscore(img)

        elif self.method == 'zscore':
            img_norm = zscore(img)

        elif self.method == 'ntc':
            well_stats = self.ntc_stats[well]
            ntc_mean, ntc_std = well_stats['mean'], well_stats['std']
            ntc_mean = np.array(ntc_mean)
            ntc_std = np.array(ntc_std)
            ntc_std += ntc_std + 1e-6
            img_norm = (img - ntc_mean[..., np.newaxis, np.newaxis]) / ntc_std[..., np.newaxis, np.newaxis]

        else:
            raise ValueError

        img_norm = img_norm.astype(np.float32)
        return img_norm


class OPSAugmentation(Augmentation):
    def __init__(
            self,
            rng: np.random.Generator = None,
            zoom_scale: float = 0.8,
            scale_std: float = 0.5,
            offset_std: float = 0.3,
            local_crop: bool = False,
            local_crop_size: int = 48,
            scale_low: float = 0.4,
            scale_upper: float = 0.6,
    ):
        super().__init__()
        self._rng = np.random.default_rng() if rng is None else rng

        if not local_crop:
            self._augmentations = ApplyListOfAugmentations(
                ApplyAugmentationRandomly(Flip.up_down(), probability=0.5, rng=self._rng),
                ApplyAugmentationRandomly(Flip.left_right(), probability=0.5, rng=self._rng),
                ApplyAugmentationRandomly(Rotate.quarter(), probability=0.5, rng=self._rng),
                AsContiguousArray(),
                Zoom(rng=self._rng, scale=zoom_scale),
                ScaleAndOffset(scale_std, offset_std, rng=self._rng),
            )
        else:
            self._augmentations = ApplyListOfAugmentations(
                ApplyAugmentationRandomly(Flip.up_down(), probability=0.5, rng=self._rng),
                ApplyAugmentationRandomly(Flip.left_right(), probability=0.5, rng=self._rng),
                ApplyAugmentationRandomly(Rotate.quarter(), probability=0.5, rng=self._rng),
                AsContiguousArray(),
                ScaleAndOffset(scale_std, offset_std, rng=self._rng),
                GenerateLocalCrops(scale_low=scale_low, scale_upper=scale_upper, resize=local_crop_size, offset=20),
            )

    def _augment_impl(self, img: np.ndarray) -> np.ndarray:
        return self._augmentations.augment(img)

    @property
    def identifier(self) -> str:
        return 'ops-augmentations'
