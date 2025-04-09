# bca/loss.py - metric classes
#
# SPDX-FileCopyrightText: Copyright (C) 2023-2024 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import medpy.metric.binary as medpyMetrics
import segmentation_models_3D as sm
from segmentation_models_3D.base import functional as F
import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

class sDSC(Metric):
  """ Dice score tensorflow metric for a single mask.

  Computes the Dice score for a single, specified output channel.
  """

  def __init__(self, channel, epsilon=0.00001, name='sDSC', **kwargs):
    """ Initialise sDSC metric.

    Args:
      * `channel`: output channel number to compute DSC for
      * `name`: metric name
      * epsilon: division by zero protection
    """
    super(sDSC, self).__init__(name=name, **kwargs)
    self.channel = channel
    self.epsilon = epsilon
    self.sdsc = self.add_weight('sdsc_v', initializer='zeros')
    self.n = self.add_weight('sdsc_n', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    """ Dice score metric for output channel (see https://arxiv.org/pdf/1606.04797.pdf).

    Dice = (2*|X&Y|+e)/ (|X|+|Y|+e) =  2*sum(|A*B|+e)/(sum(A^2)+sum(B^2)+e)
    """
    numerator = 2.0 * tf.math.reduce_sum(y_true[...,self.channel] * y_pred[...,self.channel]) + self.epsilon
    denominator = tf.math.reduce_sum(tf.math.square(y_true[...,self.channel])) + \
                  tf.math.reduce_sum(tf.math.square(y_pred[...,self.channel])) + self.epsilon
    dsc = tf.cast(tf.math.divide_no_nan(numerator, denominator), self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      sample_weight = tf.broadcast_to(sample_weight, dsc.shape)
      dsc = tf.multiply(dsc, sample_weight)
    self.sdsc.assign_add(dsc)
    self.n.assign_add(1)

  def reset_state(self):
    """ Reset metric.
    """
    self.sdsc.assign(0)
    self.n.assign(0)

  def result(self):
    """ Return metric value.
    """
    return tf.divide(self.sdsc, self.n)

  def get_config(self):
    """Get metric configuration.

    Get metric configuration required for saving/loading the models using this metric.
    """
    parent_config = super(sDSC, self).get_config()
    return {
      **parent_config,
      "channel": self.channel,
      "epsilon": self.epsilon
    }

class HD95(sm.base.Metric):
  """ Hausdorff 95 metric for evaluation only.

  Computes the Hausdorff 95th percentile distance. This is based on segmentation-models-3D
  metrics and is not a tensorflow metric, but it can be used for evaluation.
  """

  def __init__(self, name='hd95', **kwargs):
    """ Initialise hd95 metric.

    Args:
      * `name`: metric name
    """
    super(HD95, self).__init__(name=name, **kwargs)

  def __surface_distances(self, result, reference):
    """Adapted from medpy.
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    # binary structure
    footprint = generate_binary_structure(result.ndim, 1)
    # test for emptiness
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
      raise RuntimeError('No binary object.')
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    # compute average surface distance
    # Note: scipy's distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=None)
    return dt[result_border]

  def __call__(self, y_true, y_pred):
    """ HD95 metric.
    """
    try:
      hd1 = self.__surface_distances(y_pred, y_true)
      hd2 = self.__surface_distances(y_true, y_pred)
      hd95 = np.percentile(np.hstack((hd1, hd2)), 95) / 100.0
    except RuntimeError:
      hd95 = np.nan # Metric no applicable; will be ignored in stats
    return tf.convert_to_tensor(hd95)

class Specificity(sm.base.Metric):
  """ Specificity metric for evaluation only.

  This is based on segmentation-models-3D metrics and is not a tensorflow metric,
  but it can be used for evaluation.
  """

  def __init__(self, threshold=None, smooth=1e-5, name="Specificity", **kwargs):
    """ Initialise specificity metric.

    Args:
      * `threshold`: thresholding
      * `smoooth`: smoothing value
      * `name`: metric name
    """
    super(Specificity, self).__init__(name=name, **kwargs)
    self.threshold = threshold
    self.smooth = smooth

  def __call__(self, gt, pr):
    """ Specificity metric, based on segmentation-models-3D.
    """
    gt, pr = F.gather_channels(gt, pr, indexes=None, backend=tf.keras.backend)
    pr = F.round_if_needed(pr, self.threshold, backend=tf.keras.backend)
    axes = F.get_reduce_axes(False, backend=tf.keras.backend)
    tp = tf.keras.backend.sum(gt * pr, axis=axes)
    fp = tf.keras.backend.sum(pr, axis=axes) - tp
    tn = tf.keras.backend.sum((gt==0).astype(np.float32) * (pr==0).astype(np.float32), axis=axes)
    score = (tn + self.smooth)/(tn + fp + self.smooth)
    score = F.average(score, False, 1, backend=tf.keras.backend)
    return score