import numpy as np
from torchvision import transforms
# import torch
from PIL import Image
import collections
import math
import torch.nn.functional as F
RANDOM_RESOLUTIONS = [128, 256, 512, 768, 1024, 1280]


class RandomResolution(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    @staticmethod
    def get_params(img):
        w, h = img.size
        max_idx = 0
        for i in range(len(RANDOM_RESOLUTIONS)):
            if w > RANDOM_RESOLUTIONS[i] and h > RANDOM_RESOLUTIONS[i]:
                max_idx += 1
        idx = np.random.randint(max_idx)
        return idx

    def __call__(self, img):

        if self.size is None:
            idx = self.get_params(img)
            self.size = RANDOM_RESOLUTIONS[idx]
        return transforms.Resize(self.size, self.interpolation)(img)


class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)