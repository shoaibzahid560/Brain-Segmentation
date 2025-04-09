# bca/latupnet.py - LATUP-Net models
#
# SPDX-FileCopyrightText: Copyright (C) 2023-2024 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2023-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dense, Conv3D, UpSampling3D, concatenate, Activation, Multiply, Add, MaxPooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dropout, Lambda, LeakyReLU, GroupNormalization, Softmax
import tensorflow.keras.regularizers as regularizers

from .model import ModelGen

class LATUPNet(ModelGen):
  """Constructor to create 3D Latup-Net.

  This is a class to construct a network, not a Model class. It must, for any such class, contain a name field, which
  must uniquely identified the architecture specified with the parameters. It must provide a `construct` method
  which returns the tensorflow `Model`.

  Note that this class takes tensorflow objects and functions as arguments where needed, to construct the network.
  """

  def __init__(self, name, loss, attention="SE", lrelu_alpha=0.1, dropout=0.2, kernel_init="he_uniform", l2_reg=0.02,
               metrics=lambda : None, optimiser=None, fixed_batch_size=False):
    """Create an architecture class which constructs a 3D UNet as specified.
    
    Args:
      * `name`: name of the architecture, must be unique for the parameters;
      * `loss`: the loss function (e.g. as defined in `bca.trainer`)
      * `attention`: attention module used (None, SE, inception, ECA, SE3DConv, channel, spatial, CBAM, multimodal)
      * `lrelu_alpha`: LeakReLu alpha value
      * `dropout`: dropout rate
      * `kernel_init`: kernel initializer
      * `l2_reg`: L2 regularization
      * `metrics`: tensorflow metrics to record during training produced by a function / lambda expression
      * `optimiser`: tensorflow optimiser to use for training: this must be a function (e.g. lambda expression) to generate the optimiser with a single argument indicating the batch size.
      * `fixed_batch_size`: if true, use as input a fixed batch-size (may have performance advantages or sometimes needed to avoid crashes; see scw Hack in the `bca.scheduler` code)
    """
    super().__init__(name)
    self.loss = loss
    self.metrics = metrics
    self.attention = attention
    self.lrelu_alpha = lrelu_alpha
    self.dropout = dropout
    self.kernel_init = kernel_init
    self.l2_reg = l2_reg
    if optimiser is None: # Optimiser must be a function creating the optimiser using batch_size
      self.optimiser = lambda batch_size : Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    else:
      self.optimiser = optimiser
    self.fixed_batch_size = fixed_batch_size

  def construct(self, seq, batch_size, jit_compile=True):
    """Construct the tensorflow functional architecture.

    Args:
      * `seq`: keras data Sequence (see `bca.dataset.SeqGen`) for the data to be used with the model';
               this  can also be a tuple where the first part is the input size and the last tuple entry the
               number of output classes - this is to be able to generate the model independent of the data,
               e.g., for plotting.
      * `batch_size`: batch size for training; in particular this is used to adjust the learning rate for the optimiser
      * `jit_compile`: `jit_compile` argument for `Model.fit`

    Return:
      * Constructed, compiled, functional model with specified optimiser (Adam is default)
    """
    # Construct model (incl. compile)
    if isinstance(seq,tuple):
      # For plotting model, seq should be a tuple (used only by plot below)
      inputs = Input(shape=seq[:-1],batch_size=None, name='input_layer')
      num_classes = seq[-1] # Last element of tuple is number of classes
    else:
      if len(seq.cache.inp_chs) != 1 or len(seq.cache.out_chs) != 1:
        raise RuntimeError("Invalid input/output numbers")
      inputs = Input(shape=(*seq.dim,len(seq.cache.inp_chs[0])),batch_size=batch_size if self.fixed_batch_size else None, name='input_layer')
      num_classes = len(seq.cache.out_chs[0]) # Determine classes from output shape

    # Below we replaced InstanceNormalization from tensorflow-addons in the original implementation with GroupNormalization where the number of groups is the number of channels of the previous layer. This should be identical and avoids the dependency on tensorflow-addons.

    # LATUP-Net U-Net-like architecture

    # Encoder Level 1
    #  Parallel Convolution Module
    #    Embedded Layer
    x1_s = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                  padding='same', name='enc1_pc_embed')(inputs)
    #    First PC path
    x1_1 = Conv3D(32, (1, 1, 1), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                  padding='same', name='enc1_pc_1_conv')(x1_s)
    x1_1 = MaxPooling3D((2, 2, 2), name='enc1_pc_1_maxpool')(x1_1)
    #    Second PC path
    x1_2 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                  padding='same', name='enc1_pc_2_conv')(x1_s)
    x1_2 = MaxPooling3D((2, 2, 2), name='enc1_pc_2_maxpool')(x1_2)
    #    Third PC path
    x1_3 = Conv3D(32, (5, 5, 5), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                  padding='same', name='enc1_pc_3_conv')(x1_s)
    x1_3 = MaxPooling3D((2, 2, 2), name='enc1_pc_3_maxpool')(x1_3)
    #    Concatenate PC output
    x1 = concatenate([x1_1, x1_2, x1_3], axis=-1, name='enc1_pc_concat')

    # Encoder Level 2
    #  Attention layer
    x2 = self._attention(x1, "enc2")
    #  Convolution+InstanceNormalization block
    x2 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='enc2_conv1')(x2)
    x2 = GroupNormalization(groups=64, axis=-1, name='enc2_instance_norm')(x2)
    #  2nd convolution
    x2_s = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                  kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='enc2_conv2')(x2)
    #  Dropout
    x2 = Dropout(self.dropout, name='enc2_dropout')(x2_s)
    #  Max pooling
    x2 = MaxPooling3D((2, 2, 2), name='enc2_maxpool')(x2)

    # Encoder Level 3
    #  Attention layer
    x3 = self._attention(x2, "enc3")
    #  Convolution+InstanceNormalization block
    x3 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha),
                kernel_initializer=self.kernel_init, kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='enc3_conv1')(x3)
    x3 = GroupNormalization(groups=128, axis=-1, name='enc3_instance_norm')(x3)
    #  2nd convolution
    x3_s = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha),
                  kernel_initializer=self.kernel_init, kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='enc3_conv2')(x3)
    #  Dropout
    x3 = Dropout(self.dropout, name='enc3_dropout')(x3_s)
    #  Max pooling
    x3 = MaxPooling3D((2, 2, 2), name='enc3_maxpool')(x3)

    # Bottleneck
    #  Attention layer
    x4 = self._attention(x3, "bn")

    # Decoder Level 3
    #  Upsample
    y3 = UpSampling3D(size=(2,2,2), name='dec3_upsample')(x4)
    #  Convolution+InstanceNormalization block
    y3 = Conv3D(128, (2, 2, 2), activation=LeakyReLU(alpha=self.lrelu_alpha),
                kernel_initializer=self.kernel_init, padding='same', name='dec3_conv1')(y3)
    y3 = GroupNormalization(groups=128, axis=-1, name='dec3_instance_norm')(y3)
    #  Concatenate skip connection
    y3 = concatenate([y3, x3_s], name='dec3_concat')
    #  2nd convolution
    y3 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='dec3_conv2')(y3)
    #  Attention layer
    y3 = self._attention(y3, "dec3")
    #  Dropout
    y3 = Dropout(self.dropout, name='dec3_dropout')(y3)
    #  3rd convolution
    y3 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='dec3_conv3')(y3)

    # Decoder Level 2
    #  Upsample
    y2 = UpSampling3D(size =(2,2,2), name='dec2_upsample')(y3)
    #  Convolution+InstanceNormalization block
    y2 = Conv3D(64, (2, 2, 2), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                padding='same', name='dec2_conv1')(y2)
    y2 = GroupNormalization(groups=64, axis=-1, name='dec2_instance_norm')(y2)
    #  Concatenate skip connection
    y2 = concatenate([y2, x2_s], name='dec2_concat')
    #  2nd convolution
    y2 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='dec2_conv2')(y2)
    #  Attention layer
    y2 = self._attention(y2, "dec2")
    #  Dropout
    y2 = Dropout(self.dropout, name='dec2_dropout')(y2)
    #  3rd convolution
    y2 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='dec2_conv3')(y2)

    # Decoder Level 1
    #  Upsample
    y1 = UpSampling3D(size =(2,2,2), name='dec1_upsample')(y2)
    #  Convolution+InstanceNormalization block
    y1 = Conv3D(32, (2, 2, 2), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                padding='same', name='dec1_conv1')(y1)
    y1 = GroupNormalization(groups=32, axis=-1, name='dec1_instance_norm')(y1)
    #  Concatenate skip connection
    y1 = concatenate([y1,  x1_s], name='dec1_concat')
    #  2nd convolution
    y1 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='dec1_conv2')(y1)
    #  Attention layer
    y1 = self._attention(y1, "dec1")
    #  Dropout
    y1 = Dropout(self.dropout, name='dec1_dropout')(y1)
    #  3rd convolution
    y1 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=self.lrelu_alpha), kernel_initializer=self.kernel_init,
                kernel_regularizer=regularizers.l2(self.l2_reg), padding='same', name='upsample_layer_3_conv2')(y1)

    # Probability distribution map (output)
    outputs = Conv3D(num_classes, (1, 1, 1), kernel_regularizer=regularizers.l2(self.l2_reg), activation='softmax', name='prob_map')(y1)

    # Setup model
    model = Model(inputs=[inputs], outputs=[outputs], name=self.name)
    model.compile(loss=self.loss, optimizer=self.optimiser(batch_size), metrics=self.metrics(), jit_compile=jit_compile)
    return model

  # Attention layer inclusion
  def _attention(self, x, name):
    if self.attention == "SE":
      return self._att_SE(x, name)
    elif self.attention == "inception":
      return self._att_inception_module_3d(x, name)
    elif self.attention == "ECA":
      return self._att_ECA(x, name)
    elif self.attention == "SE3DConv":
      return self._att_se_3dconv(x, name)
    elif self.attention == "channel":
      return self._att_channel(x, name)
    elif self.attention == "spatial":
      return self._att_spatial(x, name)
    elif self.attention == "CBAM":
      return self._att_CBAM(x, name)
    elif self.attention == "multimodal":
      return self._att_multimodal(x, name)
    if self.attention is not None and self.attention != "None":
      raise RuntimeError(f"Unknown attention module {self.attention}")
    return x

  # Squeeze and excitation attention
  @staticmethod
  def _att_SE(x, name, ratio=8):
    c = x.shape[4]
    y = GlobalAveragePooling3D(name=f"{name}_SE_global_avg")(x)
    y = Dense(c//ratio, activation="relu", use_bias=False, name=f"{name}_SE_fc_relu")(y)
    y = Dense(c, activation="sigmoid", use_bias=False, name=f"{name}_SE_fc_sigmoid")(y)
    y = Multiply(name=f"{name}_SE_mult")([x, y])
    return y

  # Inception Module
  @staticmethod
  def _att_inception_module_3d(x, name, base_channels=32):
    # 1x1x1 convolution
    a = Conv3D(base_channels*2, (1, 1, 1), activation='relu', name=f"{name}_IN_a_conv")(x)
    # 1x1x1 followed by 3x3x3 convolution
    b = Conv3D(base_channels*4, (1, 1, 1), activation='relu', name=f"{name}_IN_b_conv1")(x)
    b = Conv3D(base_channels*4, (3, 3, 3), padding='same', activation='relu', name=f"{name}_IN_b_conv2")(b)
    # 1x1x1 followed by 5x5x5 convolution
    c = Conv3D(base_channels, (1, 1, 1), activation='relu', name=f"{name}_IN_c_conv1")(x)
    c = Conv3D(base_channels, (5, 5, 5), padding='same', activation='relu', name=f"{name}_IN_c_conv2")(c)
    # 3x3x3 max-pooling followed by 1x1x1 convolution
    d = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name=f"{name}_IN_d_maxpool")(x)
    d = Conv3D(base_channels, (1, 1, 1), activation='relu', name=f"{name}_IN_d_conv")(d)
    y = concatenate([a, b, c, d], axis=-1, name=f"{name}_IN_concat")
    return y

  # ECA attention
  @staticmethod
  def _att_ECA(x, name):
    c = x.shape[4]
    y = GlobalAveragePooling3D(name=f"{name}_ECA_global_avg")(x)
    y = Dense(c, activation="softmax", use_bias=False, name=f"{name}_ECA_fc")(y)
    y = Lambda(lambda z: tf.expand_dims(tf.expand_dims(tf.expand_dims(z, 1), 1), 1), name=f"{name}_ECA_expand")(y)
    y = Multiply(name=f"{name}_ECA_mult")([x, y])
    return y

  # Squeeze and excitation with 3D conv instead of dense layer
  @staticmethod
  def _att_se_3dconv(x, name, ratio=8):
    c = x.shape[4]
    y = GlobalAveragePooling3D(name=f"{name}_SEC_global_avg")(x)
    y = Reshape((1, 1, 1, c), name=f"{name}_SEC_reshape")(y)
    y = Conv3D(c//ratio, kernel_size=(1,1,1), activation="relu", use_bias=False, name=f"{name}_SEC_conv1")(y)
    y = Conv3D(c, kernel_size=(1,1,1), activation="sigmoid", use_bias=False, name=f"{name}_SEC_conv2")(y)
    y = Multiply(name=f"{name}_SEC_mult")([x, y])
    return y

  # Channel attention
  @staticmethod
  def _att_channel(x, name):
    c = x.shape[4]
    l1 = Dense(c, activation="sigmoid", use_bias=False, name=f"{name}_CH_fc1")
    l2 = Dense(c, use_bias=False, name=f"{name}_CH_fc2")
    # average pooling
    y1 = GlobalAveragePooling3D(name=f"{name}_CH_global_avg")(x)
    y1 = l1(y1)
    y1 = l2(y1)
    # max pooling
    y2 = GlobalMaxPooling3D(name=f"{name}_CH_global_maxpool")(x)
    y2 = l1(y2)
    y2 = l2(y2)
    # add both and apply sigmoid
    y = Add(name=f"{name}_CH_add")([y1,y2])
    y = Activation("sigmoid", name=f"{name}_CH_sigmoid")(y)
    y = Multiply(name=f"{name}_CH_mult")([x, y])
    return y

  # Spatial attention
  @staticmethod
  def _att_spatial(x, name):
    c = x.shape[4]
    # average pooling
    y1 = Lambda(lambda z : tf.expand_dims(tf.reduce_mean(z, axis=-1), axis=-1), name=f"{name}_SP_mean")(x)
    # max pooling
    y2 = Lambda(lambda z : tf.expand_dims(tf.reduce_max(z, axis=-1), axis=-1), name=f"{name}_SP_max")(x)
    # concatenate
    y = concatenate([y1 , y2], axis=-1, name=f"{name}_SP_concat")
    # conv layer
    y = Conv3D(c, kernel_size=7 ,padding='same', activation="sigmoid", name=f"{name}_SP_conv")(y)
    y = Multiply(name=f"{name}_SP_mult")([x, y])
    return y

  # CBAM attention
  @staticmethod
  def _att_CBAM(x, name):
    x = LATUPNet._att_channel(x, name+"_CBAM")
    x = LATUPNet._att_spatial(x, name+"_CBAM")
    return x

  # Multi-modal attention
  @staticmethod
  def _att_multimodal(x, name, ratio=16):
    # compute attention maps for each channel
    _, h, w, d, c = x.shape
    y = Reshape((h*w*d, c), name=f"{name}_MM_reshape1")(x)
    y = Dense(c//ratio, activation=LeakyReLU(alpha=0.1), use_bias=False, name=f"{name}_MM_fc1")(y)
    y = Dense(c, activation=LeakyReLU(alpha=0.1), use_bias=False, name=f"{name}_MM_fc2")(y)
    y = Reshape((h, w, d, c), name=f"{name}_MM_reshape2")(y)
    # sum the attention maps across channels
    y = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name=f"{name}_MM_sum")(y)
    y = Softmax(name=f"{name}_MM_softmax")(y)
    # multiply attention maps with original feature maps
    y = Multiply(name=f"{name}_MM_mult")([x, y])
    return y