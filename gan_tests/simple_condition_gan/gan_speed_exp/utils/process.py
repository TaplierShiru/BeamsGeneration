# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf
import numpy as np


COLORSPACE_RGB = 'RGB'
COLORSPACE_LAB = 'LAB'

# TF


def preprocess(img, colorspace_in, colorspace_out):
    """
    Pre proccess function for images, which normalize input images transfer to certain color space
    NOTICE! Input images should be NOT normalized.

    Parameters
    ----------
    img : tf.Tensor or tf.Variable
        Tensor/Variable of real images.
    colorspace_in : str
        Color space of input images.
        RGB color space: "RGB" string,
        LAB color space: "LAB" string,
        Register does not matter in this case
    colorspace_out : str
        Color space of output images.
        RGB color space: "RGB" string,
        LAB color space: "LAB" string,
        Register does not matter in this case
    Returns
    ----------
    tf.Tensor
        Normalized image according to `colorspace_in` and `colorspace_out`
    """
    if colorspace_out.upper() == COLORSPACE_RGB:
        if colorspace_in == COLORSPACE_LAB:
            img = lab_to_rgb(img)

        # [0, 1] => [-1, 1]
        img = (img / 255.0) * 2 - 1

    elif colorspace_out.upper() == COLORSPACE_LAB:
        if colorspace_in == COLORSPACE_RGB:
            img = rgb_to_lab(img / 255.0)

        L_chan, a_chan, b_chan = tf.unstack(img, axis=-1)

        # L: [0, 100] => [-1, 1]
        # A, B: [-110, 110] => [-1, 1]
        img = tf.stack([L_chan / 50 - 1, a_chan / 110, b_chan / 110], axis=-1)

    return img


def postprocess(img, colorspace_in, colorspace_out):
    """
    Post proccess function for images, which transfer normalize input images transfer to their color space.

    Parameters
    ----------
    img : tf.Tensor or tf.Variable
        Tensor/Variable of real images.
    colorspace_in : str
        Color space of input images.
        RGB color space: "RGB" string, If `colorspace_in` equal to RGB, `img` should be in rage [-1, 1]
        LAB color space: "LAB" string, If `colorspace_in` equal to LAB, `img` should be in rage [-1, 1]
        Register does not matter in this case
    colorspace_out : str
        Color space of output images.
        Final image will be not normalized, for example RGB image will be in range [0, 255].
        RGB color space: "RGB" string,
        LAB color space: "LAB" string,
        Register does not matter in this case
    Returns
    ----------
    tf.Tensor
        Original image according to `colorspace_in` and `colorspace_out`
    """
    if colorspace_in.upper() == COLORSPACE_RGB:
        # [-1, 1] => [0, 1]
        img = (img + 1) / 2

        if colorspace_out == COLORSPACE_LAB:
            img = rgb_to_lab(img)

    elif colorspace_in.upper() == COLORSPACE_LAB:
        L_chan, a_chan, b_chan = tf.unstack(img, axis=-1)

        # L: [-1, 1] => [0, 100]
        # A, B: [-1, 1] => [-110, 110]
        img = tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=-1)

        if colorspace_out == COLORSPACE_RGB:
            img = lab_to_rgb(img)

    return img


def rgb_to_lab(srgb):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                        ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                        xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6 / 29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                        fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                        (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def img_LAB2RGB_postprocess(img, sess=None):
    """
    Auxiliary function for restoring `img` from LAB color space to RGB.
    Input images should be normalized in their color space.
    Usually this function is used as parameter for 'restore_image_function` for training GAns

    Parameters
    ----------
    img : np.ndarray
        Array of images, which is will be translated to RGB color space from LAB
    sess : tf.Session
        Session to get result.
        By default equal to `None`,
        session will be created in this function and safely closed at the end of all operations
    """
    was_created = False
    if sess is None:
        sess = tf.Session()
        was_created = True
    img = tf.constant(img, dtype=tf.float32)
    img = postprocess(img, COLORSPACE_LAB, COLORSPACE_RGB)

    final_output = sess.run(img) * 255.0

    if was_created:
        sess.close()

    return final_output


def img_RGB2LAB_preprocess(img, sess=None):
    """
    Auxiliary function for restoring `img` from RGB color space to LAB and at the end normalize output in range [-1, 1].
    Input images should be NOT normalized.
    Usually this function is used as preprocessing function

    Parameters
    ----------
    img : np.ndarray
        Array of images, which is will be translated to LAB color space from RGB
    sess : tf.Session
        Session to get result.
        By default equal to `None`,
        session will be created in this function and safely closed at the end of all operations
    """
    was_created = False
    if sess is None:
        sess = tf.Session()
        was_created = True
    img = tf.constant(img, dtype=tf.float32)
    img = preprocess(img, COLORSPACE_RGB, COLORSPACE_LAB)

    final_output = sess.run(img)

    if was_created:
        sess.close()

    return final_output

# ----------------------------
# |          Numpy           |
# ----------------------------


def preprocess_np(img, colorspace_in, colorspace_out):
    """
    Pre proccess function for images, which normalize input images transfer to certain color space
    NOTICE! Input images should be NOT normalized.

    Parameters
    ----------
    img : np.ndarray
        Tensor/Variable of real images.
    colorspace_in : str
        Color space of input images.
        RGB color space: "RGB" string,
        LAB color space: "LAB" string,
        Register does not matter in this case
    colorspace_out : str
        Color space of output images.
        RGB color space: "RGB" string,
        LAB color space: "LAB" string,
        Register does not matter in this case
    Returns
    ----------
    np.ndarray
        Normalized image according to `colorspace_in` and `colorspace_out`
    """
    if colorspace_out.upper() == COLORSPACE_RGB:
        if colorspace_in == COLORSPACE_LAB:
            img = lab_to_rgb_np(img)

        # [0, 1] => [-1, 1]
        img = (img / 255.0) * 2 - 1

    elif colorspace_out.upper() == COLORSPACE_LAB:
        if colorspace_in == COLORSPACE_RGB:
            img = rgb_to_lab_np(img / 255.0)

        L_chan, a_chan, b_chan = img[..., 0], img[..., 1], img[..., 2]

        # L: [0, 100] => [-1, 1]
        # A, B: [-110, 110] => [-1, 1]
        img = np.stack([L_chan / 50 - 1, a_chan / 110, b_chan / 110], axis=-1)

    return img


def postprocess_np(img, colorspace_in, colorspace_out):
    """
    Post proccess function for images, which transfer normalize input images transfer to their color space.

    Parameters
    ----------
    img : np.ndarray
        Tensor/Variable of real images.
    colorspace_in : str
        Color space of input images.
        RGB color space: "RGB" string, If `colorspace_in` equal to RGB, `img` should be in rage [-1, 1]
        LAB color space: "LAB" string, If `colorspace_in` equal to LAB, `img` should be in rage [-1, 1]
        Register does not matter in this case
    colorspace_out : str
        Color space of output images.
        Final image will be not normalized, for example RGB image will be in range [0, 255].
        RGB color space: "RGB" string,
        LAB color space: "LAB" string,
        Register does not matter in this case
    Returns
    ----------
    np.ndarray
        Original image according to `colorspace_in` and `colorspace_out`
    """
    if colorspace_in.upper() == COLORSPACE_RGB:
        # [-1, 1] => [0, 1]
        img = (img + 1) / 2

        if colorspace_out == COLORSPACE_LAB:
            img = rgb_to_lab_np(img)

    elif colorspace_in.upper() == COLORSPACE_LAB:
        L_chan, a_chan, b_chan = img[..., 0], img[..., 1], img[..., 2]

        # L: [-1, 1] => [0, 100]
        # A, B: [-1, 1] => [-110, 110]
        img = np.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=-1)

        if colorspace_out == COLORSPACE_RGB:
            img = lab_to_rgb_np(img)

    return img


def rgb_to_lab_np(srgb):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    srgb_pixels = srgb.reshape(-1, 3)

    linear_mask = (srgb_pixels <= 0.04045).astype(np.float32)
    exponential_mask = (srgb_pixels > 0.04045).astype(np.float32)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    rgb_to_xyz = np.array([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ])
    xyz_pixels = np.matmul(rgb_pixels, rgb_to_xyz)

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    # normalize for D65 white point
    xyz_normalized_pixels = np.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

    epsilon = 6 / 29
    linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).astype(np.float32)
    exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).astype(np.float32)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                xyz_normalized_pixels ** (1 / 3)) * exponential_mask

    # convert to lab
    fxfyfz_to_lab = np.array([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ])
    lab_pixels = np.matmul(fxfyfz_pixels, fxfyfz_to_lab) + np.array([-16.0, 0.0, 0.0])

    return lab_pixels.reshape(*srgb.shape)


def lab_to_rgb_np(lab):
    lab_pixels = lab.reshape(-1, 3)

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    # convert to fxfyfz
    lab_to_fxfyfz = np.array([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ])
    fxfyfz_pixels = np.matmul(lab_pixels + np.array([16.0, 0.0, 0.0]), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6 / 29
    linear_mask = (fxfyfz_pixels <= epsilon).astype(np.float32)
    exponential_mask = (fxfyfz_pixels > epsilon).astype(np.float32)
    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + (
                fxfyfz_pixels ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = np.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    xyz_to_rgb = np.array([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ])
    rgb_pixels = np.matmul(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    rgb_pixels = np.clip(rgb_pixels, 0.0, 1.0)
    linear_mask = (rgb_pixels <= 0.0031308).astype(np.float32)
    exponential_mask = (rgb_pixels > 0.0031308).astype(np.float32)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                (rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return srgb_pixels.reshape(*lab.shape)


def img_LAB2RGB_postprocess_np(img, *args):
    """
    Auxiliary function for restoring `img` from LAB color space to RGB.
    Input images should be normalized in their color space.
    Usually this function is used as parameter for 'restore_image_function` for training GAns

    Parameters
    ----------
    img : np.ndarray
        Array of images, which is will be translated to RGB color space from LAB

    """
    img = postprocess_np(img, COLORSPACE_LAB, COLORSPACE_RGB)
    final_output = img * 255.0
    return final_output


def img_RGB2LAB_preprocess_np(img, *args):
    """
    Auxiliary function for restoring `img` from RGB color space to LAB and at the end normalize output in range [-1, 1].
    Input images should be NOT normalized.
    Usually this function is used as preprocessing function

    Parameters
    ----------
    img : np.ndarray
        Array of images, which is will be translated to LAB color space from RGB

    """
    img = preprocess_np(img, COLORSPACE_RGB, COLORSPACE_LAB)
    return img
