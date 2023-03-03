# -*- coding: utf-8 -*-
# @Time : 2022/4/1 11:55 上午
# @Author : ligang
# @FileName: functional_pil.py
# @Email   : ucasligang@163.com
# @Software: PyCharm
import numbers
from typing import Any, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, __version__ as PILLOW_VERSION

try:
    import accimage
except ImportError:
    accimage = None


@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@torch.jit.unused
def _get_image_size(img: Any) -> List[int]:
    if _is_pil_image(img):
        return img.size
    raise TypeError("Unexpected type {}".format(type(img)))


@torch.jit.unused
def _get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        return 1 if img.mode == 'L' else 3
    raise TypeError("Unexpected type {}".format(type(img)))


@torch.jit.unused
def hflip(img):
    """PRIVATE METHOD. Horizontally flip the given PIL Image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontally flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


@torch.jit.unused
def vflip(img):
    """PRIVATE METHOD. Vertically flip the given PIL Image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


@torch.jit.unused
def adjust_brightness(img, brightness_factor):
    """PRIVATE METHOD. Adjust brightness of an RGB image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


@torch.jit.unused
def adjust_contrast(img, contrast_factor):
    """PRIVATE METHOD. Adjust contrast of an Image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


@torch.jit.unused
def adjust_saturation(img, saturation_factor):
    """PRIVATE METHOD. Adjust color saturation of an image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


@torch.jit.unused
def adjust_hue(img, hue_factor):
    """PRIVATE METHOD. Adjust hue of an image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


@torch.jit.unused
def adjust_gamma(img, gamma, gain=1):
    r"""PRIVATE METHOD. Perform gamma correction on an image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')
    gamma_map = [(255 + 1 - 1e-3) * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


@torch.jit.unused
def pad(img, padding, fill=0, padding_mode="constant"):
    r"""PRIVATE METHOD. Pad the given PIL.Image on all sides with the given "pad" value.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple or list): Padding on each border. If a single int is provided this
            is used to pad all borders. If a tuple or list of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple or list of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively. For compatibility reasons
            with ``functional_tensor.pad``, if a tuple or list of length 1 is provided, it is interpreted as
            a single int.
        fill (int or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image: Padded image.
    """

    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, list):
        padding = tuple(padding)

    if isinstance(padding, tuple) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if isinstance(padding, tuple) and len(padding) == 1:
        # Compatibility with `functional_tensor.pad`
        padding = padding[0]

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if padding_mode == "constant":
        opts = _parse_fill(fill, img, "2.3.0", name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, **opts)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, **opts)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, tuple) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, tuple) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        p = [pad_left, pad_top, pad_right, pad_bottom]
        cropping = -np.minimum(p, 0)

        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, img.width - crop_right, img.height - crop_bottom))

        pad_left, pad_top, pad_right, pad_bottom = np.maximum(p, 0)

        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)


@torch.jit.unused
def crop(img: Image.Image, top: int, left: int, height: int, width: int) -> Image.Image:
    """PRIVATE METHOD. Crop the given PIL Image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((left, top, left + width, top + height))


@torch.jit.unused
def resize(img, size, interpolation=Image.BILINEAR):
    r"""PRIVATE METHOD. Resize the input PIL Image to the given size.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            For compatibility reasons with ``functional_tensor.resize``, if a tuple or list of length 1 is provided,
            it is interpreted as a single int.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``.

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


@torch.jit.unused
def _parse_fill(fill, img, min_pil_version, name="fillcolor"):
    """PRIVATE METHOD. Helper function to get the fill color for rotate, perspective transforms, and pad.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        fill (n-tuple or int or float): Pixel fill value for area outside the transformed
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands.
        img (PIL Image): Image to be filled.
        min_pil_version (str): The minimum PILLOW version for when the ``fillcolor`` option
            was first introduced in the calling function. (e.g. rotate->5.2.0, perspective->5.0.0)
        name (str): Name of the ``fillcolor`` option in the output. Defaults to ``"fillcolor"``.

    Returns:
        dict: kwarg for ``fillcolor``
    """
    major_found, minor_found = (int(v) for v in PILLOW_VERSION.split('.')[:2])
    major_required, minor_required = (int(v) for v in min_pil_version.split('.')[:2])
    if major_found < major_required or (major_found == major_required and minor_found < minor_required):
        if fill is None:
            return {}
        else:
            msg = ("The option to fill background area of the transformed image, "
                   "requires pillow>={}")
            raise RuntimeError(msg.format(min_pil_version))

    num_bands = len(img.getbands())
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_bands > 1:
        fill = tuple([fill] * num_bands)
    if not isinstance(fill, (int, float)) and len(fill) != num_bands:
        msg = ("The number of elements in 'fill' does not match the number of "
               "bands of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_bands))

    return {name: fill}


@torch.jit.unused
def affine(img, matrix, resample=0, fillcolor=None):
    """PRIVATE METHOD. Apply affine transformation on the PIL Image keeping image center invariant.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): image to be rotated.
        matrix (list of floats): list of 6 float values representing inverse matrix for affine transformation.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    Returns:
        PIL Image: Transformed image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    output_size = img.size
    opts = _parse_fill(fillcolor, img, '5.0.0')
    return img.transform(output_size, Image.AFFINE, matrix, resample, **opts)


@torch.jit.unused
def rotate(img, angle, resample=0, expand=False, center=None, fill=None):
    """PRIVATE METHOD. Rotate PIL image by angle.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): image to be rotated.
        angle (float or int): rotation angle value in degrees, counter-clockwise.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    Returns:
        PIL Image: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    opts = _parse_fill(fill, img, '5.2.0')
    return img.rotate(angle, resample, expand, center, **opts)


@torch.jit.unused
def perspective(img, perspective_coeffs, interpolation=Image.BICUBIC, fill=None):
    """PRIVATE METHOD. Perform perspective transform of the given PIL Image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be transformed.
        perspective_coeffs (list of float): perspective transformation coefficients.
        interpolation (int): Interpolation type. Default, ``Image.BICUBIC``.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            This option is only available for ``pillow>=5.0.0``.

    Returns:
        PIL Image: Perspectively transformed Image.
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    opts = _parse_fill(fill, img, '5.0.0')

    return img.transform(img.size, Image.PERSPECTIVE, perspective_coeffs, interpolation, **opts)


@torch.jit.unused
def to_grayscale(img, num_output_channels):
    """PRIVATE METHOD. Convert PIL image of any mode (RGB, HSV, LAB, etc) to grayscale version of image.

    .. warning::

        Module ``transforms.functional_pil`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.

    Args:
        img (PIL Image): Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns:
        PIL Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
