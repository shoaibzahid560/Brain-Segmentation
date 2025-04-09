# bca/interpret.py - interpret models
#
# SPDX-FileCopyrightText: Copyright (C) 2023-2024 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

class GradCamVisualizer:
  """GradCam visualiser.

  Visualiser for GradCam to explore impact of attention.
  """

  def __init__(self, model,seq):
    """Create GradCAM visualiser.

    Args:
      * `model`: tensorflow model.
      * `seq`: data sequence.
    """
    self.model = model
    self.seq = seq

  def visualise(self, index_image, index_slice, index_layer, layers):
    """Visualise GradCAM for a given image and layer.

    Note `index_image` is actually the batch index, but as we use batch size 1
    this corresponds to the image index. May have to change this eventually.

    Args:
      * `index_image`: index of image in sequence.
      * `index_slice`: index of slice in image.
      * `index_layer`: index of layer.
      * `layers`: list of model layers to analyse.
    """
    image, mask = self.seq[index_image]
    mask = np.argmax(mask, axis=-1)[0,:,:,:]
    prediction = self.model.predict(image)
    prediction_mask = np.argmax(prediction, axis=-1)[0,:,:,:]
    heatmap_per_class = []
    for class_idx in range(prediction.shape[-1]):
      heatmap = self._get_heatmap_layers(image, layers, class_idx)
      heatmap_per_class.append(heatmap)
    print(f"GradCAM for {index_image}-{index_slice}")
    self._plot(image, mask, prediction_mask, heatmap_per_class, index_slice, index_layer)

  def _get_heatmap_layers(self, image, list_layers, class_idx):
    # Helper to process layers for heatmap
    features_layer=[]
    for layer_name in list_layers:
      heatmap = self._make_gradcam_heatmap(image, layer_name, class_idx)
      features_layer.append(heatmap)
    return features_layer

  def _make_gradcam_heatmap(self, img_array, layer_name, class_idx):
    # Helper to compute GradCAM
    grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output])
    with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(img_array)
      loss = predictions[0,:,:,:,class_idx]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    weights = tf.reduce_mean(guided_grads, axis=(0,1,2))
    cam = np.ones(output.shape[0: 3], dtype = np.float32)
    for l, w in enumerate(weights):
      cam += w * output[:, :, :, l]
    heatmap = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return heatmap

  def _plot(self, image, mask, prediction, heatmaps, index_slice, layer_index):
    # Helper to produce GradCAM plot.

    # Adjust subplot creation for a single row
    fig, axs = plt.subplots(1,len(heatmaps)+3,sharex=True,sharey=True,dpi=Cfg.val["screen_dpi"],figsize=(Cfg.val["figsize"][0]*(len(heatmaps)+3),Cfg.val["figsize"][1]))

    class_names = ['BG', 'NCR/NET', 'ED', 'ET'] # FIXME: for now, specific to model this has been used for
    heatmap_combined = np.zeros(heatmaps[0][layer_index].shape[:2])
    for class_idx in range(len(heatmaps)):
      axs[class_idx].imshow(image[0,:,:,index_slice, 0], cmap='gray')
      axs[class_idx].imshow(heatmaps[class_idx][layer_index][:,:,index_slice], cmap='hot', alpha=0.5)
      axs[class_idx].set_title(f'{class_names[class_idx]}')
      heatmap_combined += heatmaps[class_idx][layer_index][:,:,index_slice]

    dcmap = colors.ListedColormap(['k','r','g','b']) # FIXME: for now, specific to model this has been used for

    axs[len(heatmaps)].imshow(image[0,:,:,index_slice, 0], cmap='gray')
    masked_mask = np.ma.masked_where(mask[:,:,index_slice] == 0, mask[:,:,index_slice])
    axs[len(heatmaps)].imshow(masked_mask, cmap=dcmap, alpha=0.5, vmin=0, vmax=3)
    axs[len(heatmaps)].set_title('Ground Truth')

    axs[len(heatmaps)+1].imshow(image[0,:,:,index_slice, 0], cmap='gray')
    masked_prediction = np.ma.masked_where(prediction[:,:,index_slice] == 0, prediction[:,:,index_slice])
    axs[len(heatmaps)+1].imshow(masked_prediction, cmap=dcmap, alpha=0.5, vmin=0, vmax=3)
    axs[len(heatmaps)+1].set_title('Prediction')

    axs[len(heatmaps)+2].imshow(heatmap_combined, cmap='hot', alpha=0.5)
    axs[len(heatmaps)+2].set_title("Combined")

    plt.tight_layout()
    plt.show()

class ConfusionMatrices:
  """Confusion matrix between model predictions and ground truth.
  
  Class to compute and visualise the confusion matrix between model predictions and ground truth.
  """

  def __init__(self, model, seq, class_names=['BackGround', '(NCR/NET)', '(ED)', '(ET)']):
    """Create new confusion matrix.

    Args:
      * `model`: tensorflow model.
      * `seq`: data sequence.
      * `class_names`: list of class names for classification.
    """
    self.model = model
    self.seq = seq
    self.class_names = class_names

  def get_all(self):
    """Get all confusion matrices.
    
    Note, this class removes samples/patients without an ET class to avoid inconsistent results,
    as some samples do not have ET regions.
    
    Return:
        * Pair of aggregated confusion matrix across all classes and individual confusion matrices per class.
    """
    et_class_index = self.class_names.index('(ET)')
    aggregated_cm = np.zeros([len(self.class_names), len(self.class_names)])
    conf_matrices_all = []
    for image,mask in self.seq:
      prediction = self.model.predict(image, verbose=0)
      prediction_mask = np.argmax(prediction, axis=-1)[0,:,:,:]
      mask =  np.argmax(mask, axis=-1)[0,:,:,:]
      cm = ConfusionMatrices._confusion_matrix(mask, prediction_mask)
      # Include this matrix only if 'ET' class is present in ground truth or predictions
      if et_class_index in np.unique(mask):
        conf_matrices_all.append(cm)
        aggregated_cm += cm
    return aggregated_cm, conf_matrices_all

  @staticmethod
  def _confusion_matrix(y_true, y_pred):
    # Helper to construct confusion matrix
    y_true = np.reshape(y_true, -1)
    y_pred = np.reshape(y_pred, -1)
    return confusion_matrix(y_true, y_pred)

  @staticmethod
  def calculate_mean_std(matrices):
    """Calculate mean and std across confusion matrices.

    Args:
        * `matrices`: list of matrices

    Return:
        * `mean_matrix`: mean confusion matrix.
        * `std_matrix`: std confusion matrix.
    """
    normalized_matrices = []
    for cm in matrices:
      row_sums = cm.sum(axis=1)[:, np.newaxis]
      normalized_cm = cm.astype('float') / (row_sums + 1e-10)
      normalized_matrices.append(normalized_cm)

    normalized_matrices_array = np.array(normalized_matrices)
    mean_matrix = np.mean(normalized_matrices_array, axis=0)
    std_matrix = np.std(normalized_matrices_array, axis=0)

    return mean_matrix, std_matrix

  def plot_heatmap(self, matrix, title, model_path):
    """Plot confusion matrix as heatmap.

    Args:
        * `matrix`: confusion matrix.
        * `title`: title for plot.
        * `model_path`: path to model / name.
    """
    print(f"Model: {model_path}")
    row_sums = matrix.sum(axis=1)[:, np.newaxis]
    normalized_matrix = matrix.astype('float') / (row_sums + 1e-10)
    fig = plt.figure(dpi=Cfg.val["screen_dpi"],figsize=(Cfg.val["figsize"][0],Cfg.val["figsize"][1]))
    sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap="RdBu_r", xticklabels=self.class_names, yticklabels=self.class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
