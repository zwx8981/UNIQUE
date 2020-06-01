from __future__ import division

import math
import random

import numpy as np
import torch
from numpy import linalg

try:
    import accimage
except ImportError:
    accimage = None
import numbers
from scipy import misc, ndimage
import collections
from torchvision import transforms


def _is_numpy_image(img):
    return isinstance(img, np.ndarray)


def crop(pic, i, j, h, w):
    if not _is_numpy_image(pic):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(pic)))

    return pic[i:i + h, j:j + w, :]


class BilateralFilter(object):
    def __init__(self, sigma_s=0.05, sigma_r=0.6, n_iter=5):
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.n_iter = n_iter

    def __call__(self, pic):
        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        pic = self.bilateral(pic, self.sigma_s, self.sigma_r, self.n_iter)
        return pic

    def bilateral(self, img, sigma_s, sigma_r, num_iterations, J=None):
        if img.ndim == 3:
            img = img.copy()
        else:
            h, w = img.shape
            img = img.reshape((h, w, 1))

        if J is None:
            J = img

        if J.ndim == 2:
            h, w = J.shape
            J = np.reshape(J, (h, w, 1))

        h, w, num_channels = J.shape

        dIcdx = np.diff(J, n=1, axis=1)
        dIcdy = np.diff(J, n=1, axis=0)

        dIdx = np.zeros((h, w))
        dIdy = np.zeros((h, w))

        for c in range(num_channels):
            dIdx[:, 1:] = dIdx[:, 1:] + np.abs(dIcdx[:, :, c])
            dIdy[1:, :] = dIdy[1:, :] + np.abs(dIcdy[:, :, c])

        dHdx = (1.0 + sigma_s / sigma_r * dIdx)
        dVdy = (1.0 + sigma_s / sigma_r * dIdy)

        dVdy = dVdy.T

        N = num_iterations
        F = img.copy()

        sigma_H = sigma_s

        for i in range(num_iterations):
            sigma_H_i = sigma_H * math.sqrt(3.0) * (2.0 ** (N - (i + 1))) / math.sqrt(4.0 ** N - 1.0)

            F = self.rec_filter_horizontal(F, dHdx, sigma_H_i)
            F = np.swapaxes(F, 0, 1)
            F = self.rec_filter_horizontal(F, dVdy, sigma_H_i)
            F = np.swapaxes(F, 0, 1)

        return F

    @staticmethod
    def rec_filter_horizontal(img, D, sigma):
        a = math.exp(-math.sqrt(2.0) / sigma)

        F = img.copy()
        V = np.power(a, D)

        h, w, num_channels = img.shape

        for i in range(1, w):
            for c in range(num_channels):
                F[:, i, c] = F[:, i, c] + V[:, i] * (F[:, i - 1, c] - F[:, i, c])

        for i in range(w - 2, -1, -1):
            for c in range(num_channels):
                F[:, i, c] = F[:, i, c] + V[:, i + 1] * (F[:, i + 1, c] - F[:, i, c])

        return F


class MedianFilter(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, pic):

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        pic = ndimage.median_filter(pic, size=self.size)
        return pic


class RotateImage(object):
    def __init__(self, angles):
        if isinstance(angles, tuple):
            assert len(angles) == 2, \
                'angles should be a list with the lower and upper bounds to sample the angle or a number'
            self.angles = np.float32(np.random.uniform(*angles))
        else:
            self.angles = angles

    def __call__(self, pic):
        from skimage.transform import rotate

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        rot = pic.copy()
        for index in range(3):
            channel = rot[index, :, :]
            channel = rotate(channel, self.angles, resize=False, preserve_range=True)
            rot[index, :, :] = channel
        return np.float32(rot)

    def test(self):

        from matplotlib import pyplot as plt
        pic = np.zeros((3, 10, 10))
        pic[:, 3:6, :] = 1

        rot = self(pic)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(pic.transpose((1, 2, 0)))
        axarr[1].imshow(rot.transpose((1, 2, 0)))

        plt.show()


class RandomCrop(object):
    """
    Performs a random crop in a given numpy array using only the first two dimensions (width and height)
    """

    def __init__(self, size, ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):

        # read dimensions (width, height, channels)
        w, h, c = pic.shape

        # read crop size
        th, tw = output_size

        # get crop indexes
        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)

        return i, j, th, tw

    def __call__(self, pic):
        """

        :param input: numpy array
        :return: numpy array croped using self.size
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make it three channel
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        # get crop params: starting pixels and size of the crop
        i, j, th, tw = self.get_params(pic, self.size)

        # perform cropping and return the new image
        return pic[i:i + th, j:j + tw, :]


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            pic (np array): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to the crop for center crop.
        """

        w, h, c = pic.shape
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def __call__(self, pic):
        """
        Args:
            pic (np array): Image to be cropped.
        Returns:
            np array: Cropped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        # get crop params: starting pixels and size of the crop
        i, j, h, w = self.get_params(pic, self.size)

        return pic[i:i + h, j:j + w, :]


class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop
    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.
    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        w, h = img.shape[0], img.shape[1]
        crop_h, crop_w = self.size

        if crop_w > w or crop_h > h:
            raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size, (h, w)))

        tl = crop(img, 0, 0, crop_w, crop_h)
        tr = crop(img, w - crop_w, 0, w, crop_h)
        bl = crop(img, 0, h - crop_h, crop_w, h)
        br = crop(img, w - crop_w, h - crop_h, w, h)
        center = CenterCrop(self.size)(img)

        return (tl, tr, bl, br, center)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Normalize_01(object):
    """
    Normalize the values of a numpy array between 0-1
    """

    def __init__(self, min=None, max=None):
        """

        :param min: minimum value, by default None. Useful to normalize 0-1 globally
               max: maximum value, by default None. Useful to normalize 0-1 globally
        """
        self.min = min
        self.max = max

    def __call__(self, pic):
        """
        :param pic: numpy array
        :return: same array with its values normalized between 0-1
        """
        min = self.min if self.min is not None else np.min(pic)
        max = self.max if self.max is not None else np.max(pic)

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))
        pic = (pic - min) / (max - min)
        return pic


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    Code from git repo (I do not remember which one)
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ToTensor(object):
    """
    Convert a ``numpy.ndarray`` to tensor.
    """

    def __call__(self, pic):
        """
        Args:
            converts pic (numpy array) to Tensor

        Returns:
            Tensor: Converted image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        if len(pic.shape) == 1: return torch.FloatTensor(pic.copy())

        return torch.FloatTensor(pic.transpose((2, 0, 1)).copy())


class Scale(object):
    """
    Rescale the given numpy image to a specified size.
    """

    def __init__(self, size, interpolation="bilinear"):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pic):

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        if isinstance(self.size, int):
            # if size is specified with one dimension only get the second one keeping the
            # aspect-ratio

            # get the size of the original image
            w, h = pic.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return pic

            # calculate the ouput size keeping the aspect-ratio
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)

            # create the output array
            img_out = np.zeros((ow, oh, pic.shape[2]))

            if len(pic.shape) == 3:
                # if 3D image, scale each channel individually
                for i in range(pic.shape[2]):
                    img_out[:, :, i] = misc.imresize(pic[:, :, i], (ow, oh), interp=self.interpolation, mode='F')
                return img_out
            else:
                # if 2D image, scale image
                return misc.imresize(pic, (ow, oh), interp=self.interpolation, mode='F')
        else:
            # if size is specified with 2 dimensions apply the scale directly
            # create the output array

            if len(pic.shape) == 3:
                img_out = np.zeros((self.size[0], self.size[1], pic.shape[2]))

                # if 3D image, scale each channel individually
                for i in range(pic.shape[2]):
                    img_out[:, :, i] = misc.imresize(pic[:, :, i], self.size, interp=self.interpolation, mode='F')
                return img_out
            else:
                # if 2D image, scale image
                return misc.imresize(pic, self.size, interp=self.interpolation, mode='F')


class DownScale(object):
    """
    Smart Downscale the given numpy image to a specified size.
    """

    def __init__(self, size, interpolation="bilinear"):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pic):

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # get the size of the original image
        w, h = pic.shape[:2]
        if w <= self.size or h <= self.size:
            return pic

        # calculate the ouput size keeping the aspect-ratio
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)

        # create the output array
        img_out = np.zeros((ow, oh, pic.shape[2]))

        if len(pic.shape) == 3:
            # if 3D image, scale each channel individually
            for i in range(pic.shape[2]):
                img_out[:, :, i] = misc.imresize(pic[:, :, i], (ow, oh), interp=self.interpolation, mode='F')
            return img_out
        else:
            # if 2D image, scale image
            return misc.imresize(pic, (ow, oh), interp=self.interpolation, mode='F')


class rgb2xyz(object):
    """
    Transform a numpy array in the form [H, W, C] from RGB color space to XYZ color space.
    """

    def __init__(self):
        self.matrix = np.array([[0.412453, 0.357580, 0.180423],
                                [0.212671, 0.715160, 0.072169],
                                [0.019334, 0.119193, 0.950227]])

    def __call__(self, pic):
        """

        :param input: numpy array in RGB color space
        :return: numpy array in XYZ color space
        """
        if isinstance(pic, np.ndarray):
            # from skimage import color
            # return color.rgb2lab(pic, self.illuminant, self.observer)

            arr = np.asanyarray(pic)

            if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
                msg = ("the input array must be have a shape == (.., ..,[ ..,] 3)), " +
                       "got (" + (", ".join(map(str, arr.shape))) + ")")
                raise ValueError(msg)

            return np.dot(arr, self.matrix.T.copy())

            # out_img = np.zeros(pic.shape)
            #
            # for row in range(pic.shape[0]):
            #     for col in range(pic.shape[1]):
            #         out_img[row, col] = np.dot(self.matrix, pic[row, col])
            #
            # return out_img
        else:
            raise TypeError("Tensor [pic] is not numpy array")


class xyz2rgb(object):
    def __init__(self):
        self.matrix = linalg.inv(rgb2xyz().matrix)

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # from skimage import color
            # return color.rgb2lab(pic, self.illuminant, self.observer)

            arr = np.asanyarray(pic)

            if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
                msg = ("the input array must be have a shape == (.., ..,[ ..,] 3)), " +
                       "got (" + (", ".join(map(str, arr.shape))) + ")")
                raise ValueError(msg)

            return np.dot(arr, self.matrix.T.copy())

            # out_img = np.zeros(pic.shape)
            #
            # for row in range(pic.shape[0]):
            #     for col in range(pic.shape[1]):
            #         out_img[row, col] = np.dot(self.matrix, pic[row, col])
            #
            # return out_img
        else:
            raise TypeError("Tensor [pic] is not numpy array")


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a probability of 0.5."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make it three channel
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        if random.random() < self.prob:
            return pic[:, ::-1, :]
        return pic


class RandomVerticalFlip(object):
    """Vertically flip the given numpy array randomly with a probability of 0.5 by default."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, pic):
        """
        Args:
            img (numpy array): Image to be flipped.
        Returns:
            numpy array: Randomly flipped image.
        """

        # check type of [pic]
        if not _is_numpy_image(pic):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make it three channel
        if len(pic.shape) != 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        if random.random() < self.prob:
            return pic[::-1, :, :]
        return pic


class Lambda(transforms.Lambda):
    pass


class Compose(transforms.Compose):
    pass


class Normalize(transforms.Normalize):
    pass
