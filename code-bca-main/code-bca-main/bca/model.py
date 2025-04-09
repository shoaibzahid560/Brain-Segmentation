# bca/model.py - Abstract model class
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import tensorflow as tf

from abc import ABC, abstractmethod

class ModelGen(ABC):
  """Abstract base class for model generators.

  This is a class is the base class for classes to construct a model. It must, for any such class, contain
  a name field, which must uniquely identified the architecture specified with the parameters. It must provide
  a `construct` method which returns the tensorflow `Model`. The plot method should be universal. The
  constructor should set he parameters for the `construct` method.
  """

  @abstractmethod
  def __init__(self, name):
    """Abstract base class method to create an architecture class which constructs a model as specified.

    For an example of how to use it, see the `Unet3D` class in `bca/unet.py`.
    
    Args:
      * `name`: name of the architecture, must be unique for the parameters;
    """
    self.name = name

  @abstractmethod
  def construct(self, seq, batch_size, jit_compile=True):
    """Construct a tensorflow functional architecture.

    Args:
      * `seq`: keras data Sequence (see `bca.dataset.SeqGen`) for the data to be used with the model';
               this  can also be a tuple where the first part is the input size and the last tuple entry the
               number of output classes - this is to be able to generate the model independent of the data,
               e.g., for plotting.
      * `batch_size`: batch size for training; in particular this is used to adjust the learning rate for the optimizer
      * `jit_compile`: `jit_compile` argument for `Model.fit`

    Return:
      * Constructed and compiled model with optimizer
    """
    pass

  def plot(self, dim, text=False):
    """Plot model architecture.
    
    It runs in a separate process such that GPU resources used are cleared after the run. If the summary text/plot
    files already exist, it does not recreate them (recall that model names must be unique). Of course you can
    delete the cached files.

    Args:
      * `dim`: Model input shape - spatial dimensions and number of channels (last)
      * `text`: If True, show text summary instead.
    """
    # Run this in separate process so we clear resources afterwards
    import os
    file = self.__class__.__name__+"plot.png" # temporary file
    if Cfg.val['multiprocessing']:
      import multiprocessing
      try:
        p = multiprocessing.Process(target=self._plot,kwargs={"file": file, "dim": dim, "text": text})
        p.start()
        p.join()
      except:
        p.kill()
      if p.exitcode != 0:
        raise RuntimeError("Process failed")
    else:
      self._plot(file, dim, text)
    if text:
      with open(file,"r") as f:
        print(f.read())
    else:
      from IPython import display
      display.display(display.Image(filename=os.path.join(file)))
    os.remove(file)

  def _plot(self, file, dim, text):
    # Get model
    model = self.construct(seq=dim, batch_size=4)
    # Plot
    if text:
      with open(file,"w") as f:
        model.summary(print_fn=lambda l : print(l, file=f), line_length=128, expand_nested=True, show_trainable=True)
    else:
      tf.keras.utils.plot_model(model, to_file=file, show_shapes=True, 
                                show_dtype=True, show_layer_names=True, show_layer_activations=True,
                                rankdir='TB', expand_nested=True, dpi=Cfg.val['screen_dpi'])
