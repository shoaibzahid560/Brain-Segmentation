# bca/loss.py - loss classes
#
# SPDX-FileCopyrightText: Copyright (C) 2023-2024 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss

def compute_channel_weights(ds, mask_channel, channels, mode="enet", normalise=False, enet_c=1.02, no_cache=False):
  """Compute channel weights based on inverse relative volume, normalised.

  Args:
    * `ds`: dataset to compute class weights for, specifically for BraTS data
    * `mask_channel`: channel name for mask
    * `channels`: Ordered dictionary of output channels where each channel is represented as a pair `(a,b)` where `a` is a list of labels defining the output channel in the mask and `b` is a Boolean indicating if it should be included in the weight calculation; weight is set to 0 for channels that are not included in the weight calculation
    * `mode`: weight computation mode ("enet" or "inverse")
    * `normalise`: Boolean whether to normalise weight
    * `enet_c`: c constant for Enet weights
    * `no_cache`: do not use cached results if True

  Returns:
    * `weights`: dictionary of (weight,voxel_count) for channels
  """
  # Cache for voxel counts from dataset (we only use it, do not update it; that's for the dataset class)
  if ds.crops_type == "f":
    data_cache = "f"+"x".join([str(c[0])+"_"+str(c[1]) for c in ds.crops])
  elif ds.crops_type == "bb" or ds.crops_type == "orig":
    data_cache = ds.crops_type
  else:
    raise RuntimeError(f"Unknown crop type {ds.crops_type}")
  cache = os.path.join(ds.cache, f"voxel_counts_{data_cache}.csv")
  voxel_labels = [None] * len(ds)
  voxel_counts = [None] * len(ds)
  if not no_cache and os.path.isfile(cache):
    with open(cache, "r", encoding="utf-8") as f:
      rows = csv.reader(f)
      for row in rows:
        try:
          idx = ds.patients.index(row[0])
          labels = []
          counts = []
          for col in range(1,len(row)):
            l, cnts = row[col].split(":")
            labels.append(int(l))
            counts.append(int(cnts))
          voxel_labels[idx] = labels
          voxel_counts[idx] = counts
        except ValueError:
          pass
  # Get the number of pixels per class
  counts = np.zeros(5,dtype=np.int64)
  for pidx in range(0,len(ds)):
    if voxel_labels[pidx] is not None and voxel_counts[pidx] is not None:
      labels = voxel_labels[pidx]
      label_counts = voxel_counts[pidx]
    else:
      labels, label_counts = np.unique(ds.cropped(pidx,channels=[mask_channel])[mask_channel].get_fdata(), return_counts=True)
      labels = [int(l) for l in labels]
    for n, l in enumerate(labels):
      counts[l] += label_counts[n]
  # Compute the channel volume based on the counts for the labels specified to be in the channel
  channel_volumes = {}
  total_volume = 0
  for ch in channels:
    channel_volumes[ch] = np.sum(counts[l] for l in channels[ch][0]) if channels[ch][1] else 0
    total_volume += channel_volumes[ch]
  weights = {}
  if mode == "inverse":
    # Inverse relative volume
    for ch in channels:
      weights[ch] = float(total_volume)/float(channel_volumes[ch]) if channels[ch][1] else 0.0
  elif mode == "enet":
    for ch in channels:
      weights[ch] = 1/np.log(enet_c + float(channel_volumes[ch])/float(total_volume)) if channels[ch][1] else 0.0
  else:
    raise RuntimeError(f"Unknown class_weight mode {mode}")
  if normalise:
    # Compute total weight for normalisation
    total_weight = 0.0
    for ch in channels:
      total_weight += weights[ch]
    # Compute normalised weights
    for ch in channels:
      weights[ch] = weights[ch]/total_weight
  res = {}
  for ch in channels:
    res[ch] = (weights[ch], channel_volumes[ch])
  return res

class WeightedDiceLoss(Loss):
  """Weighted Dice loss class with smoothing factor.

  Compute a weighted average of the Dice loss per channel. Multiple
  channels can be merged for this as well.
  """

  def __init__(self, weights, epsilon=0.00001, **kwargs):
    """Initialise weighted Dice loss.

    Args:
      * weights: List of (weight,channel) pairs to compute the loss over;
                 channel can be a list of channels that are combined to a
                 single mask by averaging
      * epsilon: division by zero protection
    """
    super(WeightedDiceLoss, self).__init__(**kwargs)
    self.epsilon = epsilon
    self.weights = weights

  def _dice_score(self, y_true, y_pred, ch):
    """ Dice score for class specified by channel ch (see https://arxiv.org/pdf/1606.04797.pdf).

    Dice = (2*|X&Y|+e)/ (|X|+|Y|+e) =  2*(sum(|A*B|)+e)/(sum(A^2)+sum(B^2)+e)
    """
    if isinstance(ch, list):
      # Channel index is a list of indices, so we take the average over the output channels for the Dice score
      yt_avg = tf.math.add_n([y_true[:,:,:,:,c] for c in ch]) / len(ch)
      yp_avg = tf.math.add_n([y_true[:,:,:,:,c] for c in ch]) / len(ch)
      numerator = 2.0 * tf.math.reduce_sum(yt_avg * yp_avg) + self.epsilon
      denominator = tf.math.reduce_sum(tf.math.square(yt_avg)) + \
                      tf.math.reduce_sum(tf.math.square(yp_avg)) + self.epsilon
    else:
      # Dice score for single channel
      numerator = 2.0 * tf.math.reduce_sum(y_true[:,:,:,:,ch] * y_pred[:,:,:,:,ch]) + self.epsilon
      denominator = tf.math.reduce_sum(tf.math.square(y_true[:,:,:,:,ch])) + \
                      tf.math.reduce_sum(tf.math.square(y_pred[:,:,:,:,ch])) + self.epsilon
    return tf.math.divide_no_nan(numerator, denominator)

  def call(self, y_true, y_pred):
    """Computes the weighted Dice loss between the two arguments.
    """
    # Compute weighted Dice loss for each (weight,channel) pair in weights
    loss = 0.0
    for wc in self.weights:
      if wc[0] != 0.0: # Channel can be excluded via weight (see channel weight computation)
        loss += wc[0] * (1.0 - self._dice_score(y_true, y_pred, wc[1]))
    return loss

  def get_config(self):
    """Get layer configuration.

    Get layer configuration required for saving/loading the models using this layer.
    """
    parent_config = super(WeightedDiceLoss, self).get_config()
    return {
      **parent_config,
      "epsilon": self.epsilon,
      "weights": self.weights
    }

class WeightedDiceScore(Loss):
  """Weighted Dice score loss class with smoothing factor.

  Computes w0 - sum_c (w_c * DSC_c) instead of the usual Dice score,
  where c is the output channel (or Dice score of the average of 
  the output channels if c is a list).
  """

  def __init__(self, weight0, weights, epsilon=0.00001, **kwargs):
    """Initialise weighted Dice score.

    Args:
      * weight0: Constant weight
      * weights: List of (weight,channel) pairs to compute the Dice score over;
                 channel can be a list of channels that are combined to a single
                 mask by averaging
      * epsilon: division by zero protection for Dice score
    """
    super(WeightedDiceScore, self).__init__(**kwargs)
    self.epsilon = epsilon
    self.weights = weights
    self.weight0 = weight0

  def _dice_score(self, y_true, y_pred, ch):
    """Dice score for class specified by channel ch (see https://arxiv.org/pdf/1606.04797.pdf).

    Dice = (2*|X&Y|+e)/ (|X|+|Y|+e) =  2*(sum(|A*B|)+e)/(sum(A^2)+sum(B^2)+e)
    """
    if isinstance(ch, list):
      # Channel index is a list of indices, so we take the average over the output channels for the Dice score
      yt_avg = tf.math.add_n([y_true[:,:,:,:,c] for c in ch]) / len(ch)
      yp_avg = tf.math.add_n([y_true[:,:,:,:,c] for c in ch]) / len(ch)
      numerator = 2.0 * tf.math.reduce_sum(yt_avg * yp_avg) + self.epsilon
      denominator = tf.math.reduce_sum(tf.math.square(yt_avg)) + \
                      tf.math.reduce_sum(tf.math.square(yp_avg)) + self.epsilon
    else:
      # Dice score for single channel
      numerator = 2.0 * tf.math.reduce_sum(y_true[:,:,:,:,ch] * y_pred[:,:,:,:,ch]) + self.epsilon
      denominator = tf.math.reduce_sum(tf.math.square(y_true[:,:,:,:,ch])) + \
                      tf.math.reduce_sum(tf.math.square(y_pred[:,:,:,:,ch])) + self.epsilon
    return tf.math.divide_no_nan(numerator, denominator)

  def call(self, y_true, y_pred):
    """Computes the weighted Dice loss between the two arguments.
    """
    # Compute weighted Dice score for each (weight,channel) pair in weights
    loss = self.weight0
    for wc in self.weights:
      if wc[0] != 0.0: # Channel can be excluded via weight (see channel weight computation)
        loss -= wc[0] * self._dice_score(y_true, y_pred, wc[1])
    return loss

  def get_config(self):
    """Get layer configuration.

    Get layer configuration required for saving/loading the models using this layer.
    """
    parent_config = super(WeightedDiceScore, self).get_config()
    return {
      **parent_config,
      "epsilon": self.epsilon,
      "weights": self.weights,
      "weight0": self.weight0
    }
