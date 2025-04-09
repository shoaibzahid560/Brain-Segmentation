# bca/unet.py - UNet models
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, MaxPooling3D, LeakyReLU, concatenate
from tensorflow.keras.regularizers import l2

from .model import ModelGen

class UNet3D(ModelGen):
  """Constructor to create 3D UNets based on the arguments provided.

  This is a class to construct a network, not a Model class. It must, for any such class, contain a name field, which
  must uniquely identified the architecture specified with the parameters. It must provide a `construct` method
  which returns the tensorflow `Model`.

  Note that this class takes tensorflow objects and functions as arguments where needed, to construct the network.
  """

  def __init__(self, name, enc, dec, loss, classify_kernel_regularizer=l2(0.02), metrics=None, optimiser=None, fixed_batch_size=False):
    """Create an architecture class which constructs a 3D UNet as specified.
    
    Args:
      * `name`: name of the architecture, must be unique for the parameters;
      * `enc`: a list of dictionaries that are arguments for `_enc_block` method to construct the encoder;
      * `dec`: a list of dictionaries that are arguments for `_dec_block` method to construct the decoder;
      * `loss`: the loss function (e.g. as defined in `bca.trainer`)
      * `classify_kernel_regularizer`: regularizer for the final classification layer
      * `metrics`: tensorflow metrics to record during training
      * `optimiser`: tensorflow optimiser to use for training: this must be a function (e.g. lambda expression) to generate the optimiser with a single argument indicating the batch size.
      * `fixed_batch_size`: if true, use as input a fixed batch-size (may have performance advantages or sometimes needed to avoid crashes; see scw Hack in the `bca.scheduler` code)

    **`_enc_block`** is used to create an encoder block and its arguments are provided in the `enc` dictionary list:
      * `x` - input from the previous encoder block or from `Input`, using tensorflow's functional API
      * `name` - name of the encoder block (used to name its layers)
      * `filters` - number of filters
      * `kernel` - 3D convolution kernel size
      * `activation` - activation function (used as `activation` argument to `Conv3D`)
      * `kernel_initializer` - initialiser for kernel (used as `kernel_initializer` argument to `Conv3D`)
      * `dropout` - dropout rate (only if positive; otherwise layer is skipped)
      * `max_pooling` - max pooling layer; if `None`, it is not used (for last layer)
    Return:
      * If `max_pooling` is `None`:
        * Output of last layer ("before `max-pooling`"), None 
      * Otherwise:
        * max-pooling output, output before max pooling (for cross-link)

    **`_dec_block`** is used to create a decoder block and its arguments are provided in the `dec` dictionary list:
      * `x` - input from the previous decoder or the latent representation, using tensorflow's functional API
      * `y` - input linking from the corresponding encoder block, using tensorflow's functional API
      * `name` - name oft he decoder block (used to name its layers)
      * `conv_trans_filters` - number of 3D transposed-convolution filters
      * `conv_trans_strides` - 3D transposed-convolution strides
      * `filters` - number of 3D convolution filters
      * `kernel` - kernel of 3D convolutions
      * `activation` - activation function (used as `activation` argument to `Conv3D`)
      * `kernel_initializer` - initialiser for kernel (used as `kernel_initializer` argument to `Conv3D`)
      * `dropout` - dropout rate (only if positive; otherwise layer is skipped)
    Return:
      * Output of last layer

    Example:
    ```python
    from bca.trainer import dsc_loss, dsc, iou
    model = UNet3D(name="UNet3D-UniqueNameForParameters",
                   enc=[{"filters": 16},
                        {"filters": 32},
                        {"filters": 64},
                        {"filters":128, "kernel_regularizer":keras.regularizers.l2(0.02)},
                        {"filters":256, "kernel_regularizer":keras.regularizers.l2(0.02), "max_pooling":None}],
                   dec=[{"filters":128},
                        {"filters": 64},
                        {"filters": 32, "kernel_regularizer":keras.regularizers.l2(0.02)},
                        {"filters": 16, "kernel_regularizer":keras.regularizers.l2(0.02)}],
                   loss=dsc_loss,
                   metrics=[dsc,iou])
    ```
    """
    super().__init__(name)
    self.enc = enc
    self.dec = dec
    self.classify_kernel_regularizer = classify_kernel_regularizer
    self.loss = loss
    self.metrics = metrics
    if optimiser is None: # Optimiser must be a function creating the optimiser using batch_size
      self.optimiser = lambda batch_size : Adam(learning_rate = 1e-4 * batch_size / 16.0)
    else:
      self.optimiser = optimiser
    self.fixed_batch_size = fixed_batch_size

  def construct(self, seq, batch_size, jit_compile=True):
    """Construct a tensorflow functional architecture.

    Args:
      * `seq`: keras data Sequence (see `bca.dataset.SeqGen`) for the data to be used with the model';
               this  can also be a tuple where the first part is the input size and the last tuple entry the
               number of output classes - this is to be able to generate the model independent of the data,
               e.g., for plotting.
      * `batch_size`: batch size for training; in particular this is used to adjust the learning rate for the optimiser
      * `jit_compile`: `jit_compile` argument for `Model.fit`

    Retrun:
      * Constructed, compiled, functional model with specified optimiser (Adam is default)
    """
    # Construct model (incl. compile)
    if isinstance(seq,tuple):
      # For plotting model, seq should be a tuple (used only by plot below)
      inputs = Input(shape=seq[:-1],batch_size=None) 
      classes = seq[-1] # Last element of tuple is number of classes
    else:
      if len(seq.cache.inp_chs) != 1 or len(seq.cache.out_chs) != 1:
        raise RuntimeError("Invalid input/output numbers")
      inputs = Input(shape=(*seq.dim,len(seq.cache.inp_chs[0])),batch_size=batch_size if self.fixed_batch_size else None)
      classes = len(seq.cache.out_chs[0]) # Determine classes from output shape
    # Encoder
    x = inputs
    u = [None] * len(self.enc)
    for c,e in enumerate(self.enc):
      x, u[c] = UNet3D._enc_block(x, name=f"enc{c+1}", **e)
    # Decoder
    for c,d in enumerate(self.dec):
      x = UNet3D._dec_block(x, u[-c-2], name=f"dec{c+1}", **d)
    # Classifier
    x = Conv3D(classes, (1, 1, 1), kernel_regularizer=self.classify_kernel_regularizer, activation='sigmoid', name="class")(x)
    # Setup model
    model = Model(inputs=inputs, outputs=x, name=self.name) 
    model.compile(loss=self.loss, optimizer=self.optimiser(batch_size), metrics=self.metrics, jit_compile=jit_compile)
    return model

  @staticmethod
  def _enc_block(x, name, filters, kernel=(3,3,3), activation=LeakyReLU(alpha=0.1),
                 kernel_initializer='he_uniform', kernel_regularizer=None, dropout=0.2, 
                 max_pooling=(2,2,2)):
    # Encoder block (see __init__)
    x = Conv3D(filters, kernel, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
               padding='same', name=name+"-conv1")(x)
    if dropout > 0.0:
      x = Dropout(dropout, name=name+"-dropout")(x)
    x = Conv3D(filters, kernel, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
               padding='same', name=name+"-conv2")(x)
    if max_pooling is None:
      return x, None
    y = MaxPooling3D(max_pooling, name=name+"-max_pooling")(x)
    return y, x

  @staticmethod
  def _dec_block(x, y, name, filters, conv_trans_filters=(2,2,2), conv_trans_strides=(2,2,2), kernel=(3,3,3),
                 activation=LeakyReLU(alpha=0.1), kernel_initializer='he_uniform', kernel_regularizer=None,
                 dropout=0.2):
    # Decoder block (see __init__)
    x = Conv3DTranspose(filters, conv_trans_filters, strides=conv_trans_strides, padding='same', name=name+"-conv_trans")(x)
    x = concatenate([x,y])
    x = Conv3D(filters, kernel, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
               padding='same', name=name+"-conv1")(x)
    if dropout > 0.0:
      x = Dropout(dropout, name=name+"-dropout")(x)
    x = Conv3D(filters, kernel, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
               padding='same', name=name+"-conv2")(x)
    return x
