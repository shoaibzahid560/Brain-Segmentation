# bca/trainer.py - Train models
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import os
import shutil
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_3D as sm
import multiprocessing

from .loss import WeightedDiceLoss, WeightedDiceScore
from .metric import sDSC, HD95, Specificity
from .interpret import ConfusionMatrices, GradCamVisualizer

import tensorflow as tf
tf.get_logger().setLevel(Cfg.val['tf_log'])

from IPython import display

dsc = sm.metrics.FScore(name="DSC")
"""F1/Dice score metric."""
dsc_loss = sm.losses.DiceLoss()
"""F1/Dice loss."""
hd95 = HD95(name="HD95")
"""F1/Dice score metric."""
sensitivity = sm.metrics.Recall(name="Sensitivity")
"""Sensitivity metric."""
specificity = Specificity(name="Specificity")
"""Specificity metric."""
iou = sm.metrics.IOUScore(name="IoU")
"""IoU(Intersection of Union)/Jaccard index metric."""

class ChkptLogger(tf.keras.callbacks.Callback):
  """Log history and store best/last model Callback.

  This callback is used by the training to record the model history and save the best and last model.
  """  

  def __init__(self, path, best_monitor=("loss","loss"), best_cmp="less", save_traces=False, save_last=True):
    """Create a ChkptLogger.

    Args:
      * `path`: Path where to store data (folder containing the models);
      * `best_monitor`: Selects metric to decide upon the best model. This is a pair of strings where the metric name
        must start with the first entry and finish with the second (for multiple outputs, etc. and to decide between
        test or training metric);
      * `best_cmp`: Name to indicate comparison function for selecting best model (whether greater ("greater") or smaller ("less") values are better);
      * `save_traces`: Boolean indicating whether to save traces with the model;
      * `save_last`: Boolean indicating whether to save last model
    """
    super().__init__()
    self.path = path
    # History Log
    self.history_file = "history.json"
    self.history = {}
    # Model checkpoint
    self.best_model_path = os.path.join(path,"best")
    self.best_model_path_old = os.path.join(path,"best-old")
    self.best_model_path_new = os.path.join(path,"best-new")
    self.best_monitor = best_monitor # pair - first part matches start and last part matches end, to account for multiple outputs
    if best_cmp == "less":
      import numpy as np
      self.best_cmp = np.less
    elif best_cmp == "greater":
      import numpy as np
      self.best_cmp = np.grater
    else:
      raise RuntimeError(f"Unknown best_cmp {best_cmp}")
    self.last_model_path = os.path.join(path,"last")
    self.last_model_path_new = os.path.join(path,"last-new")
    self.save_traces = save_traces
    self.save_last = save_last

  def on_train_begin(self, logs=None):
    """Called at the beginning of training to process history and setup model checkpoints.

    Args:
      * `logs`: Dict. Currently no data is passed to this argument for this method but that may change in the future.
    """
    # History log
    hf = os.path.join(self.best_model_path, self.history_file)
    if os.path.exists(hf):
      with open(hf, "r") as f:
        self.history = json.load(f)
    else:
      self.history["time"] = []
      self.history["epochs"] = 0
      self.history["best"] = None
      self.history["best_epoch"] = None
      self.history["logs"] = {}
    # Model checkpoint
    for p in [self.best_model_path_old, self.best_model_path_new, self.last_model_path_new]:
      if os.path.isdir(p):
        shutil.rmtree(p)

  def on_epoch_begin(self, epoch, logs=None):
    """Called at the start of an epoch to initiate timing history.

    Args:
      * `epoch`: Integer, index of epoch';
      * `logs`: Dict. Currently no data is passed to this argument for this method but that may change in the future.
    """
    self.last_time = tf.timestamp().numpy()

  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch to record logs and update best saved model.

    Args: 
      * `epoch`: Integer, index of epoch;
      * `logs`: Dict, metric results for this training epoch, and for the validation epoch if validation is performed.
         Validation result keys are prefixed with val_. For training epoch, the values of the Model's metrics are returned.\
         Example: `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # History log
    self.history["time"].append(tf.timestamp().numpy()-self.last_time)
    logs = logs or {}
    for k in logs.keys():
      if k not in self.history["logs"]:
        self.history["logs"][k] = [np.nan]*self.history["epochs"]
      self.history["logs"][k].append(logs[k])
    for k in self.history["logs"].keys():
      if k not in logs.keys():
        self.history["logs"][k].append(np.nan)
    self.history["epochs"] += 1
    # Save best model
    cur_val = 0
    for k in logs.keys():
      # Combine values to find best, if there are multiple outputs
      if k[:len(self.best_monitor[0])] == self.best_monitor[0] and k[-len(self.best_monitor[1]):] == self.best_monitor[1]:
        cur_val += logs[k]
    if self.history["best"] is None or self.best_cmp(cur_val, self.history["best"]):
      self.history["best"] = cur_val
      self.history["best_epoch"] = self.history["epochs"]
      self.model.save(self.best_model_path_new, overwrite=False, save_traces=self.save_traces)
      with open(os.path.join(self.best_model_path_new,self.history_file), "w") as fp:
        print(json.dumps(self.history, indent=2, sort_keys=True), file=fp)
      if os.path.isdir(self.best_model_path):
        os.rename(self.best_model_path, self.best_model_path_old)
      os.rename(self.best_model_path_new, self.best_model_path)
      if os.path.isdir(self.best_model_path_old):
        shutil.rmtree(self.best_model_path_old)

  def on_train_end(self, logs=None):
    """Called at the end of training to save last model.

    Args:
      * `logs`: Dict. Currently the output of the last call to `on_epoch_end()` is passed to this argument for this method but that may change in the future.
    """
    if self.save_last:
      # Save last model
      self.model.save(self.last_model_path_new, overwrite=False, save_traces=self.save_traces)
      with open(os.path.join(self.last_model_path_new,self.history_file), "w") as fp:
        print(json.dumps(self.history, indent=2, sort_keys=True), file=fp)
      if os.path.isdir(self.last_model_path):
        shutil.rmtree(self.last_model_path)
      os.rename(self.last_model_path_new, self.last_model_path)

class Trainer:
  """Train and evaluate networks.

  This class handles training of a network, either locally by running the training directly, or
  remotely, but creating tasks to be executed by `bca.scheduler.schedule`. The training can be
  interrupted and continued, locally or remotely (it uses the ChkptLogger to store the best model
  so far and continues from it).
  """

  def __init__(self, model, epochs):
    """Create a new trainer.
    
    Args:
      * `model`: model object providing a `model.name` name variable and a `model.construct(seq, batch_size, jit_compile)`
         method to create the model. See, for instance, `bca.unet.Unet3D`.
      * `epochs`: number of training epochs.
    """
    self.model = model
    self.epochs = epochs

  def _model_path(self, seq):
    # Return path where to store the model
    return os.path.join(seq.cache.data_folder,
                        self.model.name
                        +"-"+"_".join([l for ic in seq.cache.inp_chs for l in ic])
                        +"-"+"_".join([l for oc in seq.cache.out_chs for l in oc]),
                        str(self.epochs)+"-"+str(seq.batch_size),
                        str(seq.seed)+"-"+str(seq.k)+"-"+str(seq.k_n))

  def _status(self, path):
    # Report status: start -> training:EXECUTOR -> end[done|failed]
    s_fn = os.path.join(path,"status")
    if os.path.isfile(s_fn):
      with open(s_fn, "r") as f:
        status = f.read()
      if status == "end":
        if os.path.isdir(os.path.join(path,"last")):
          return "done"
        return "failed"
      else:
        return "training"
    return "start"

  def _set_status(self, path, status):
    # Set status: start -> training:EXECUTOR -> end[done|failed]
    s_fn = os.path.join(path,"status")
    if status == "start":
      if os.path.isfile(s_fn):
        os.remove(s_fn)
      if os.path.isdir(os.path.join(path,"last")):
        os.rmtree(os.path.join(path,"last"))
    else:
      os.makedirs(path,exist_ok=True)
      with open(s_fn, "w") as f:
        f.write(status)
    return status

  def train(self, seqs, jit_compile=True, remote=False, best_monitor=("loss","loss"), best_cmp="less"):
    """Train the model.

    Trains the model on `seqs` Sequences (see `bca.dataset.SeqGen`, created from `bca.dataset.Dataset`),
    either directly on the local system or remotely by only generating the tasks to be executed by
    `bca.scheduler.schedule`. Training can be interrupted and continued from the last saved best model.
    We run this in a separate process such that after it is done the process quits and returns the
    GPU/tensorflow resources, not to occupy those while the notebook is running, etc., which can cause
    resource issues (e.g. if some other GPU process is run while the notebook is active).

    Args:
      * `seqs`: A list of keras Sequences as data generator (`bca.dataset.SeqGen`); each element of the list
        is assumed to be a pair of a training and test sequence (set the 2nd to None for no test sequence).
      * `jit_compile`: `jit_compile` argument for `Model.fit()`
      * `remote`: run jobs remotely; this means only the `task.py` files will be created for execution by the
         scheduler.
      * `best_monitor`: Selects metric to decide upon the best model. This is a pair of strings where the metric name
        must start with the first entry and finish with the second (for multiple outputs, etc. and to decide between
        test or training metric);
      * `best_cmp`: Function to compare metric values for selecting best model (whether greater or smaller values are better).
    """
    # Run this in separate process so we clear resources afterwards
    if Cfg.val["multiprocessing"]:
      try:
        p = multiprocessing.Process(target=self._train, kwargs={"seqs": seqs, "jit_compile": jit_compile,
                                                                "remote": remote, "best_monitor": best_monitor,
                                                                "best_cmp": best_cmp})
        p.start()
        p.join()
      except:
        p.kill()
      if p.exitcode != 0:
        raise RuntimeError("Process failed")
    else:
      self._train(seqs, jit_compile, remote, best_monitor, best_cmp)

  def _train(self, seqs, jit_compile, remote, best_monitor, best_cmp):
    # Train model on sequences
    if remote:
      s_start = 0
      s_training = 0
      s_done = 0
      s_failed = 0
    for k,seq in enumerate(seqs):
      print(f"* Fold {k+1}: ", end="")
      path = self._model_path(seq[0])
      train_run = False
      s = self._status(path)
      if s == "start":
        if remote:
          # Remote train model
          print(f"start remote - {path}")
          self._remote_train_model(path, seq[0], seq[1], jit_compile, best_monitor, best_cmp)
          s_start += 1
        else:
          print(f"start local - {path}")
          self._train_model(path, seq[0], seq[1], jit_compile, best_monitor, best_cmp)
          s = self._status(path)
      elif s == "training":
        if remote:
          print(f"training remote - {path}")
          s_training += 1
        else:
          # If we train locally and this is in training, it failed and we need to restart
          s_fn = os.path.join(path,"status")
          with open(s_fn, "r") as f:
            status = f.read()
          if len(status) == 8:
            # Local training (no executor specified)
            print(f"restart local - {path}")
            self._train_model(path, seq[0], seq[1], jit_compile, best_monitor, best_cmp)
            s = self._status(path)
            train_run = True
          else:
            print(f"setup remotely")
      elif s == "done":
        print(f"done - {path}")
        if remote:
          s_done += 1
      elif s == "failed":
        print(f"failed - {path}")
        if remote:
          s_failed += 1
      else:
        raise RuntimeError(f"Unknonw status {s}")
      if not remote and train_run:
        display.clear_output(wait=True)
    if remote:
      print(f"=> Done: {s_done}; Failed: {s_failed}; Training: {s_training}; Start: {s_start}")

  def _train_model(self, path, train_seq, test_seq, jit_compile, best_monitor, best_cmp):
    # Train model locally, not using any executors
    if os.path.exists(os.path.join(path, "task.py")):
      os.remove(os.path.join(path, "task.py"))
    self._set_status(path,"training")
    # In case of custom training loops or different distribution strategies, this part of the code needs to be adapted
    # We suggest to add flags to the model and adjust the code here depending on the flags; also needs to be done in _remote_train_model to match
    with tf.distribute.MirroredStrategy().scope():
      if os.path.isdir(os.path.join(path,"best")):
        # Load model checkpoint
        model = tf.keras.models.load_model(os.path.join(path,"best"), custom_objects=Trainer.custom_objects)
        with open(os.path.join(path,"best","history.json"), "r") as f:
          history = json.load(f)
        red_epochs = history["epochs"]
      else:
        # Setup model
        model = self.model.construct(seq=train_seq, batch_size=train_seq.batch_size, jit_compile=jit_compile)
        red_epochs = 0
    # Fit model
    model.fit(train_seq, validation_data=test_seq, epochs=self.epochs-red_epochs,
              callbacks=[ChkptLogger(path, best_monitor=best_monitor, best_cmp=best_cmp)], verbose=1)
    #
    self._set_status(path,"end")

  def _remote_train_model(self, path, train_seq, test_seq, jit_compile, best_monitor, best_cmp):
    # Training file is not fully secure and must be created by task scheduler for remote tasks (depends on platform)
    # Create task script
    os.makedirs(path,exist_ok=True)
    best_model_path = os.path.join(path,'best')
    if not os.path.isdir(best_model_path):
      # Save initial model so it can be loaded by task
      model = self.model.construct(seq=train_seq, batch_size=train_seq.batch_size, jit_compile=jit_compile)
      # We need to call this to initialise the optimizer such that the script can load the model for training
      # Setting the gradient to 0 should not change the trainable variables ?
      model.optimizer.apply_gradients(zip([0.0]*len(model.trainable_weights), model.trainable_weights))
      #
      model.save(best_model_path, overwrite=True, save_traces=False)
      history = {
          "time": [],
          "epochs": 0,
          "best": None,
          "best_epoch": None,
          "logs": {}
        }
      with open(os.path.join(best_model_path,"history.json"), "w") as fp:
        print(json.dumps(history, indent=2, sort_keys=True), file=fp)
      tf.keras.backend.clear_session()
    task_file = os.path.join(path, "task.py")
    with open(task_file, "w") as f:
      f.write(f"# Training task: {path}\n")
      f.write( "import os\n")
      f.write( "import json\n")
      f.write(f"for dir in {Cfg.val['xla_gpu_cuda_data_path']}:\n")
      f.write( "  if os.path.isdir(dir):\n")
      f.write( '    os.environ["XLA_FLAGS"]=f"--xla_gpu_cuda_data_dir={dir}"\n')
      f.write( "    break\n")
      f.write( "import numpy as np\n")
      f.write( "import tensorflow as tf\n")
      f.write(f"tf.get_logger().setLevel('{Cfg.val['tf_log']}')\n")
      f.write( "from bca.trainer import ChkptLogger, Trainer\n")
      f.write( "from bca.dataset import Cache, SeqGen\n")
      f.write(f"cache = Cache('{train_seq.cache.data_folder}', {train_seq.cache.channels}, {train_seq.cache.dim}, {train_seq.cache.inp_chs}, {train_seq.cache.out_chs}, '{train_seq.cache.seg_mask}')\n")
      f.write(f"ns_train={train_seq.names}\n")
      f.write(f"train_seq = SeqGen(ns_train, cache, {train_seq.dim}, {train_seq.batch_size}, k={train_seq.k}, k_n={train_seq.k_n}, seed={train_seq.seed}, shuffle={train_seq.shuffle})\n")
      f.write(f"ns_test={test_seq.names}\n")
      f.write(f"test_seq = SeqGen(ns_test, cache, {test_seq.dim}, {test_seq.batch_size}, k={test_seq.k}, k_n={test_seq.k_n}, seed={test_seq.seed}, shuffle={test_seq.shuffle})\n")
      f.write(f"best_path=os.path.join('{path}','best')\n")
      f.write( "with open(os.path.join(best_path,'history.json'), 'r') as f:\n")
      f.write( "  history = json.load(f)\n")
      f.write( "red_epochs = history['epochs']\n")
      # In case of custom training loops or different distribution strategies, this part of the code needs to be adapted
      # We suggest to add flags to the model and adjust the code here depending on the flags; also needs to be done in _train_model to match
      f.write( "with tf.distribute.MirroredStrategy().scope():\n")
      f.write( "  model = tf.keras.models.load_model(best_path, custom_objects=Trainer.custom_objects)\n")
      f.write(f"model.fit(train_seq, validation_data=test_seq, epochs={self.epochs}-red_epochs, callbacks=[ChkptLogger('{path}', best_monitor={best_monitor}, best_cmp='{best_cmp}')], verbose=1)\n")

  def eval(self, seqs, mode="best", fs=[dsc,iou], std_eval=None):
    """Evaluate models for Sequences.

    This creates the `evaluation.json` file in the model folder with the evaluation results for the
    trained model. If the file already exists, it is assumed the evaluation is complete.

    It runs in a separate process such that GPU resources used are cleared after the run.

    Args:
      * `seqs`: A list of keras Sequences as data generator (`bca.dataset.SeqGen`); each element of the list
        is assumed to be a pair of a training and test sequence (set the 2nd to None for no test sequence).
      * `mode`: "last" or "best" to decide which model to evaluate on (we generally assume "best")
      * `fs`: Metrics to use (must be per sample, but called on single sample with sample index)
      * `std_eval`: Function mapping single output P and expected output Y sample onto standardised data to make metrics comparable; metrics are also computed for this output; output should be a dictionary mapping NAMES to standardised lists [PP,YY]; generally we assume NAMES linked to labels for BraTS2020: "whole": 1,2,4, "necrotic": 1, "enhancing": 4, "edema": 2, "core": 1,4 but any can be used (for space reasons we often abbreviate to three letters). The function needs to know how to convert the output of the network to the standardised data. Note that P and Y are lists of tensors, for each output tensor, even if there is only one output, while PP and YY should only be one output tensor to be compared with the metrics. Generally we have P[OUTPUT-INDEX][SAMPLE=0,data-axes,OUTPUT-CHANNEL] and PP[SAMPLE=0,data-axes,STD_OUTPUT-CHANNEL] (often only one STD_OUTPUT-CHANNEL).
    """
    # Run this in separate process so we clear resources afterwards
    if Cfg.val["multiprocessing"]:
      try:
        p = multiprocessing.Process(target=self._eval, kwargs={"seqs": seqs, "mode": mode, "fs": fs,
                                                               "std_eval": std_eval})
        p.start()
        p.join()
      except:
        p.kill()
      if p.exitcode != 0:
        raise RuntimeError("Process failed")
    else:
      self._eval(seqs, mode, fs, std_eval)

  def _eval(self, seqs, mode, fs, std_eval):
    # Evaluate the sequences; helper to run it in separate process
    for k,seq in enumerate(seqs):
      print(f"* Fold {k+1}")
      self._eval_model(seq[0], seq[1], mode, fs, std_eval)
    print("Evaluation complete.")
    display.clear_output(wait=True)

  def _eval_model(self, train_seq, test_seq, mode, fs, std_eval):
    # Eval model for train/test sequence
    # FIXME: this needs to be sped up by not processing each sample individually
    path = self._model_path(train_seq)
    # Return if model does not exist
    if not os.path.isdir(os.path.join(path,"last")):
      print("Model did not complete training")
      return
    eval_path = os.path.join(path,mode,"evaluation.json")
    if os.path.isfile(eval_path):
      # Check if evaluation is OK or needs updating/is broken
      with open(eval_path, "r") as f:
        eval_data = json.load(f)
      broken = False
      for f in ["train_total", "test_total", "train_per_sample", "test_per_sample", 
                "train_std_per_sample", "test_std_per_sample"]:
        if f not in eval_data or len(eval_data[f]) == 0:
          broken = True
          break
      if not broken:
        return
      os.remove(eval_path)

    # Get model
    model = tf.keras.models.load_model(os.path.join(path,mode), custom_objects=Trainer.custom_objects)

    # Evaluate model on sequence with model metrics
    tre = model.evaluate(x=train_seq, verbose=1)
    if test_seq is None:
      tee = {}
    else:
      tee = model.evaluate(x=test_seq, verbose=1)
    # Map to dictionary, with fixing issue if there is only one metric
    try:
      tre = dict(zip(model.metrics_names, tre))
      tee = dict(zip(model.metrics_names, tee))
    except:
      tre = dict(zip(model.metrics_names, [tre]))
      tee = dict(zip(model.metrics_names, [tee]))

    # Evaluate prediction of each train/test sample with metrics specified in fs
    # Train predictions
    print("Evaluating training samples...")
    tr_res = {} # Direct/raw evaluation data for training
    tr_std = {} # Standardised evaluation data for training
    for X, Y in train_seq:
      # Collect metrics per sample
      if not isinstance(Y,list):
        Y = [Y]
      P = model.predict(X, verbose=0)
      if not isinstance(P,list):
        P = [P]
      # Metrics on actual output
      for k in range(0,P[0].shape[0]):
        for kk in range(0,len(Y)):
          for f in fs:
            key = f.name
            if len(Y) > 1:
              key += "-"+str(kk) # Add output index, if multiple outputs
            val = float(f(Y[kk][k:k+1,...], P[kk][k:k+1,...]).numpy())
            if not np.isnan(val): # nan means metric not applicable, so do not include in stats
              if key in tr_res:
                tr_res[key].append(val)
              else:
                tr_res[key] = [val]
            # Last channel is index for output maps - if there is more than one
            # map we apply the metrics to each individually as well to handle
            # multi-segmentation, etc. results
            if Y[kk].shape[-1] > 1:
              for ch in range(0, Y[kk].shape[-1]):
                ch_key = key + f"_c{ch}"
                val = float(f(Y[kk][k:k+1,...,ch], P[kk][k:k+1,...,ch]).numpy())
                if ch_key in tr_res:
                  tr_res[ch_key].append(val)
                else:
                  tr_res[ch_key] = [val]
      # Standardised metrics
      if std_eval is not None:
        for k in range(0,P[0].shape[0]):
          # Standardise data
          std_data = std_eval([P[kk][k:k+1,...] for kk in range(0,len(Y))], 
                              [Y[kk][k:k+1,...] for kk in range(0,len(Y))])
          for std_name in std_data:
            # Compute metrics for standardised data
            if std_name not in tr_std:
              tr_std[std_name] = {}
            for f in fs:
              key = f.name
              val_std = float(f(std_data[std_name][0], std_data[std_name][1]).numpy())
              if not np.isnan(val_std): # nan means metric not applicable, so do not include in stats
                if key in tr_std[std_name]:
                  tr_std[std_name][key].append(val_std)
                else:
                  tr_std[std_name][key] = [val_std]
    # Test predictions
    te_res = {} # Direct/raw evaluation data for testing
    te_std = {} # Standardised  evaluation data for testing
    if test_seq is not None:
      print("Evaluating test samples...")
      for X, Y in test_seq:
        # Collect metrics per sample
        if not isinstance(Y,list):
          Y = [Y]
        P = model.predict(X, verbose=0)
        if not isinstance(P,list):
          P = [P]
        for k in range(0,P[0].shape[0]):
          for kk in range(0,len(Y)):
            # Metrics on actual output
            for f in fs:
              key = f.name
              if len(Y) > 1:
                key += "-"+str(kk) # Add output index, if multiple outputs
              val = float(f(Y[kk][k:k+1,...],P[kk][k:k+1,...]).numpy())
              if not np.isnan(val): # nan means metric not applicable, so do not include in stats
                if key in te_res:
                  te_res[key].append(val)
                else:
                  te_res[key] = [val]
              # Last channel is index for output maps - if there is more than one
              # map we apply the metrics to each individually as well to handle
              # multi-segemtnation, etc. results
              if Y[kk].shape[-1] > 1:
                for ch in range(0, Y[kk].shape[-1]):
                  ch_key = key + f"_c{ch}"
                  val = float(f(Y[kk][k:k+1,...,ch], P[kk][k:k+1,...,ch]).numpy())
                  if ch_key in te_res:
                    te_res[ch_key].append(val)
                  else:
                    te_res[ch_key] = [val]
        # Standardised metrics
        if std_eval is not None:
          for k in range(0,P[0].shape[0]):
            # Standardise data
            std_data = std_eval([P[kk][k:k+1,...,] for kk in range(0,len(Y))], 
                                [Y[kk][k:k+1,...] for kk in range(0,len(Y))])
            for std_name in std_data:
              # Compute metrics for standardised data
              if std_name not in te_std:
                te_std[std_name] = {}
              for f in fs:
                key = f.name
                val_std = float(f(std_data[std_name][0], std_data[std_name][1]).numpy())
                if not np.isnan(val_std): # nan means metric not applicable, so do not include in stats
                  if key in te_std[std_name]:
                    te_std[std_name][key].append(val_std)
                  else:
                    te_std[std_name][key] = [val_std]

    # Save evaluation data
    eval = {
        "train_total": tre, 
        "test_total": tee,
        "train_per_sample": tr_res,
        "test_per_sample": te_res,
        "train_std_per_sample": tr_std,
        "test_std_per_sample": te_std
      }
    with open(eval_path, "w") as fp:
      print(json.dumps(eval, indent=2, sort_keys=True), file=fp)

    # Cleanup
    tf.keras.backend.clear_session()

  def get_eval(self, seqs, mode):
    """Get evaluation data.

    This loads the `evaluation.json` file in the model folder with the evaluation results for the
    trained models. It fails if these have not been generated.

    Args:
      * `seqs`: A list of keras Sequences as data generator (`bca.dataset.SeqGen`); each element of the list
        is assumed to be a pair of a training and test sequence (set the 2nd to None for no test sequence).
      * `mode`: "last" or "best" to decide which model to evaluate on (we generally assume "best")
    """
    res = []
    for k,seq in enumerate(seqs):
      print(f"* Fold {k+1}")
      res.append(self._get_eval_model(seq[0], seq[1], mode))
    return res

  def _get_eval_model(self, train_seq, test_seq, mode):
    # Get evaluation data for model train/test sequence
    path = self._model_path(train_seq)
    # Return if model does not exist
    if not os.path.isdir(os.path.join(path,"last")):
      raise RuntimeError("Model did not complete training")
    eval_path = os.path.join(path,mode,"evaluation.json")
    if not os.path.isfile(eval_path):
      raise RuntimeError("No evaluation data present")
    # Check if evaluation is OK or needs updating/is broken
    with open(eval_path, "r") as f:
      eval_data = json.load(f)
    for f in ["train_total", "test_total", "train_per_sample", "test_per_sample", 
              "train_std_per_sample", "test_std_per_sample"]:
      if f not in eval_data or len(eval_data[f]) == 0:
        raise RuntimeError("Evaluation data is incomplete")
    return eval_data

  def interpreter(self, seq, mode="best", index_image=0, index_layer=0, index_slice=0, vis='CM', layers=None):
    """Interpret the model based on the data in the given sequence.

    This creates the `evaluation.json` file in the model folder with the evaluation results for the
    trained model. If the file already exists, it is assumed the evaluation is complete.

    It runs in a separate process such that GPU resources used are cleared after the run.

    Args:
      * `seq`: Data sequence to interpret.
      * `mode`: "best" or "last" model to use for interpretation
      * `index_image`: Sample index in sequence to visualise (only for "HM")
      * `index_slice`: Slice index for sample to visualise (only for "HM")
      * `index_layer`: Layer index (only for "HM")
      * `vis`: Interpretation visualisation: "CM" for confusion matrix; "HM": for GradCam heatmap
      * `layers`: list of model layers to analyse (only for "HM")
    """
    # Run this in separate process so we clear resources afterwards
    if Cfg.val["multiprocessing"]:
      try:
        p = multiprocessing.Process(target=self._interpreter, kwargs={"seq": seq,
                                                                      "mode": mode,
                                                                      "index_image": index_image,
                                                                      "index_slice": index_slice,
                                                                      "index_layer": index_layer,
                                                                      "vis": vis,
                                                                      "layers": layers})
        p.start()
        p.join()
      except:
        p.kill()
      if p.exitcode != 0:
        raise RuntimeError("Process failed")
    else:
      self._interpreter(seq, mode, index_image, index_slice, index_layer, vis, layers)
  
  def _interpreter(self, seq, mode, index_image, index_slice, index_layer, vis, layers):
    # Helper method for interpreter multiprocessing
    # FIXME: should really make this an interactive plot or something like this, but for now this is sufficient.
    model_path = self._model_path(seq) # Assuming seq is a single data sequence
    # Return if model does not exist
    if not os.path.isdir(os.path.join(model_path,"last")):
      print("Model did not complete training")
      return
    if not os.path.isdir(model_path):
      print("Model not found")
      return
    # Get model
    model_path = os.path.join(model_path, mode)
    model = tf.keras.models.load_model(model_path, custom_objects=Trainer.custom_objects)

    if vis == 'CM':
      # Visualise confusion matrices
      cm = ConfusionMatrices(model=model, seq=seq)
      _, all_cf = cm.get_all()
      mn, std = ConfusionMatrices.calculate_mean_std(all_cf)
      cm.plot_heatmap(mn, 'Mean Confusion Matrix', model_path)
      cm.plot_heatmap(std, 'Std Deviation Confusion Matrix', model_path)
    elif vis=='HM':
      # Visualise heatmap
      gc_vis = GradCamVisualizer(model=model, seq=seq)
      gc_vis.visualise(index_image, index_slice, index_layer, layers)
    else:
      raise RuntimeError(f"Unknown visualisation mode {vis}")

    # Cleanup
    tf.keras.backend.clear_session()

  def plot_model(self, seq, mode="best", text=False, save_only=False):
    """Plot model.
    
    The model must exist to have the shapes, so this is for analysis after training even if not strictly needed.

    It runs in a separate process such that GPU resources used are cleared after the run. If the summary text/plot
    files already exist, it does not recreate them (recall that model names must be unique). Of course you can
    delete the cached files.

    Args:
      * `seqs`: A list of keras Sequences as data generator or a single Sequence from the generators. This is used
        to determine which model to load and plot from all the options (in principle they should all be the same,
        but sometimes they may not). If a list is provided the `seq[0][0]` model is used.
      * `mode`: "last" or "best" to decide which model to evaluate on (we generally assume "best")
      * `text`: If True, show text summary instead.
      * `save_only`: If True, only save files, do not show anything
    """
    # Run this in separate process so we clear resources afterwards
    if isinstance(seq, list):
      seq = seq[0][0]
    path = self._model_path(seq)
    im_dpi=True
    for dpi in Cfg.val['image_dpi']:
      if not os.path.isfile(os.path.join(path,"..","architecture@"+str(dpi)+".png")):
        im_dpi=False
        break
    if not (os.path.isfile(os.path.join(path,"..","summary.txt")) and im_dpi and            
            os.path.isfile(os.path.join(path,"..","architecture@"+str(Cfg.val['screen_dpi'])+".png"))):
      if Cfg.val["multiprocessing"]:
        try:
          p = multiprocessing.Process(target=self._plot_model,kwargs={"path": path, "seq": seq, "mode": mode})
          p.start()
          p.join()
        except:
          p.kill()
        if p.exitcode != 0:
          raise RuntimeError("Process failed")
      else:
        self._plot_model(path, seq, mode)
    if not save_only:
      if text:
        with open(os.path.join(path,"..","summary.txt"),"r") as f:
          print(f.read())
      else:
        display.display(display.Image(filename=os.path.join(path,"..","architecture@"+str(Cfg.val['screen_dpi'])+".png")))

  def _plot_model(self, path, seq, mode="best"):
    # Get model
    if os.path.isdir(os.path.join(path,mode)):
      model = tf.keras.models.load_model(os.path.join(path,mode), custom_objects=Trainer.custom_objects)
    else:
      os.makedirs(path,exist_ok=True)
      model = self.model.construct(seq=seq, batch_size=seq.batch_size)
    # Plot
    with open(os.path.join(path,"..","summary.txt"),"w") as f:
      model.summary(print_fn=lambda l : print(l, file=f), line_length=128, expand_nested=True, show_trainable=True)
    for dpi in [Cfg.val["screen_dpi"], *Cfg.val["image_dpi"]]:
      tf.keras.utils.plot_model(model, to_file=os.path.join(path,"..","architecture@"+str(dpi)+".png"), show_shapes=True, 
                                show_dtype=True, show_layer_names=True, show_layer_activations=True,
                                rankdir='TB', expand_nested=True, dpi=dpi)

  def plot_results(self, seqs, eval_mode="best"):
    """Plot model histories and analysis results after training and evaluation is complete.

    This collects the evaluation results for the sequences and plots them for a jupyter notebook.

    The x' prime values are the averages of the per-sample metrics for each fold.

    Args:
      * `seqs`: A list of keras Sequences as data generator or a single Sequence from the generators. This is used
        to determine which model to load and plot from all the options (in principle they should all be the same,
        but sometimes they may not). If a list is provided the `seq[0][0]` model is used.
      * `mode`: "last" or "best" to decide which model to evaluate on (we generally assume "best")

    Return:
      * pandas frame of training results over the folds
    """
    history = []
    evals = []
    for l in range(len(seqs)):
      path = self._model_path(seqs[l][0])
      if not os.path.isdir(os.path.join(path,"last")):
        print(f"Stopping evaluation: model for fold {l} did not complete training")
        return
      if not os.path.isfile(os.path.join(path,eval_mode,"evaluation.json")):
        print(f"{eval_mode} model for fold {l} not evaluated")
        return
      with open(os.path.join(path, "last", "history.json"), "r") as f:
        history.append(json.load(f))
      with open(os.path.join(path, eval_mode, "evaluation.json"), "r") as f:
        evals.append(json.load(f))

    # Plot histories
    fig_cols = max(4,len(history))
    fig = plt.figure(dpi=Cfg.val["screen_dpi"],figsize=(Cfg.val["figsize"][0]*fig_cols,Cfg.val["figsize"][1]*(3+len(evals[0]["train_per_sample"]))))
    std_eval_len = len(evals[0]["train_std_per_sample"])
    if std_eval_len > 0:
      std_eval_len *= len(evals[0]["train_std_per_sample"][list(evals[0]["train_std_per_sample"].keys())[0]])
    gs = fig.add_gridspec(3+len(evals[0]["train_per_sample"])+std_eval_len,fig_cols)
    loss_n = len([k for k in history[0]["logs"] if "loss" in k])
    cols = plt.cm.rainbow(np.linspace(0,1,len(history[0]["logs"])))
    ax0 = None
    for k in range(0,len(history)):
      if k == 0:
        ax = fig.add_subplot(gs[0,k])
        ax.set_ylabel("Loss")
        ax0 = ax
      else:    
        ax = fig.add_subplot(gs[0,k], sharex=ax0, sharey=ax0)
      ax.set_xlabel(f"Epoch (Fold {k+1})")
      ax.set_prop_cycle(color=cols[0:loss_n])
      ax2 = None
      for key in history[k]["logs"].keys():
        if "loss" in key:
          ax.plot(range(1,len(history[k]["logs"][key])+1), history[k]["logs"][key], label=key)
        else:
          if ax2 == None:
            ax2 = ax.twinx()
            ax2.set_prop_cycle(color=cols[loss_n:])
            if k == len(history)-1:
              ax2.set_ylabel("Metric")
          ax2.plot(range(1,len(history[k]["logs"][key])+1), history[k]["logs"][key], label=key)
      if ax2 is not None:
        li1, la1 = ax.get_legend_handles_labels()
        li2, la2 = ax2.get_legend_handles_labels()
        ax.legend(li1+li2, la1+la2, loc="lower left", bbox_to_anchor=(-0.12,0.99), ncol=3)
      else:
        ax.legend(li, la)

    # Plot times per epoch
    ax0 = None
    for k in range(0,len(history)):
      ax = fig.add_subplot(gs[1,k])
      sns.histplot(data=history[k]["time"],color='#1f77b4')
      m = np.nanmean(history[k]["time"])
      s = np.nanstd(history[k]["time"])
      plt.axvline(x=m,color='#1f77b4')
      plt.errorbar(x=m,y=np.mean(ax.get_ylim()),xerr=s,color='#1f77b4')
      ax.set_xlabel(f"Time (s) per Epoch: {m:.4f}σ{s:.4f}")

    # Metric distributions over samples
    palette = plt.cm.rainbow(np.linspace(0,1,2*len(evals[0]["train_per_sample"].keys()))).tolist()
    palette = np.array(palette[0::2] + palette[1::2])
    for k in range(0,len(evals)):
      for f,fk in enumerate(evals[k]["train_per_sample"].keys()):
        # Raw network metrics

        ax = fig.add_subplot(gs[2+f,k])
        # Train

        mean = np.nanmean(evals[k]["train_per_sample"][fk])
        std = np.nanstd(evals[k]["train_per_sample"][fk])
        sns.histplot(data=evals[k]["train_per_sample"][fk],
                     label=f"Train {eval_mode} {fk}': {mean:.4f}σ{std:.4f}",
                     ax=ax,color=palette[2*f])
        plt.sca(ax)
        plt.axvline(x=mean,color=palette[2*f])
        r = ax.get_ylim()
        ax.errorbar(x=mean,y=np.mean(r)+0.02*(r[1]-r[0]),xerr=std,color=palette[2*f])
        # Test
        if evals[k]["test_per_sample"] is not None:
          mean = np.nanmean(evals[k]["test_per_sample"][fk])
          std = np.nanstd(evals[k]["test_per_sample"][fk])
          sns.histplot(data=evals[k]["test_per_sample"][fk],
                       label=f"Test {eval_mode} {fk}': {mean:.4f}σ{std:.4f}",
                       ax=ax,color=palette[2*f+1])
          plt.sca(ax)
          plt.axvline(x=mean,color=palette[2*f+1])
          r = ax.get_ylim()
          ax.errorbar(x=mean,y=np.mean(r)-0.02*(r[1]-r[0]),xerr=std,color=palette[2*f+1])
        ax.legend()

    # Standardised metrics
    for k in range(0,len(evals)):
      for keyn, key in enumerate(evals[k]["train_std_per_sample"]):
        for f,fk in enumerate(evals[0]["train_std_per_sample"][key].keys()):
          ax = fig.add_subplot(gs[2+len(evals[k]["train_per_sample"].keys())
                                  +keyn*len(evals[0]["train_std_per_sample"][key].keys())+f,k])
          # Train
          mean = np.nanmean(evals[k]["train_std_per_sample"][key][fk])
          std = np.nanstd(evals[k]["train_std_per_sample"][key][fk])
          sns.histplot(data=evals[k]["train_std_per_sample"][key][fk],
                      label=f"Train {eval_mode} STD {key.upper()} {fk}: {mean:.4f}σ{std:.4f}",
                      ax=ax,color=palette[2*f])
          plt.sca(ax)
          plt.axvline(x=mean,color=palette[2*f])
          r = ax.get_ylim()
          ax.errorbar(x=mean,y=np.mean(r)+0.02*(r[1]-r[0]),xerr=std,color=palette[2*f])
          # Test
          if evals[k]["test_std_per_sample"] is not None:
            mean = np.nanmean(evals[k]["test_std_per_sample"][key][fk])
            std = np.nanstd(evals[k]["test_std_per_sample"][key][fk])
            sns.histplot(data=evals[k]["test_std_per_sample"][key][fk],
                         label=f"Test {eval_mode} STD {key.upper()} {fk}: {mean:.4f}σ{std:.4f}",
                         ax=ax,color=palette[2*f+1])
            plt.sca(ax)
            plt.axvline(x=mean,color=palette[2*f+1])
            r = ax.get_ylim()
            ax.errorbar(x=mean,y=np.mean(r)-0.02*(r[1]-r[0]),xerr=std,color=palette[2*f+1])
          ax.legend()

    # Results
    data = np.zeros((len(seqs)+2, len(evals[0]["train_total"]) + len(evals[0]["test_total"]) +
                                  len(evals[0]["train_per_sample"]) + len(evals[0]["test_per_sample"]) +
                                  std_eval_len +
                                  (std_eval_len if len(evals[0]["test_per_sample"]) > 0 else 0) ))
    cs=[]
    idx=[]
    for k in range(0,len(seqs)):
      idx.append(f"Fold {k+1}")
      r = 0
      # Overall metrics
      for c,key in enumerate(evals[k]["train_total"].keys()):
        data[k,r] = evals[k]["train_total"][key]
        r += 1
        if k == 0:
          cs.append(f"{key}")
        if key in evals[k]["test_total"]:
          data[k,r] = evals[k]["test_total"][key]
          r += 1
          if k == 0:
            cs.append(f"val_{key}")
      # Per-sample metrics
      for c,key in enumerate(evals[k]["train_per_sample"].keys()):
        data[k,r] = np.nanmean(evals[k]["train_per_sample"][key])
        r += 1
        if k == 0:
          cs.append(f"{key}'")
        if key in evals[k]["test_per_sample"]:
          data[k,r] = np.nanmean(evals[k]["test_per_sample"][key])
          r += 1
          if k == 0:
            cs.append(f"val_{key}'")
      # STD per-sample metrics
      for c,key in enumerate(evals[k]["train_std_per_sample"].keys()):
        for cc,metrickey in enumerate(evals[k]["train_std_per_sample"][key].keys()):
          data[k,r] = np.nanmean(evals[k]["train_std_per_sample"][key][metrickey])
          r += 1
          if k == 0:
            cs.append(f"STD-{key}-{metrickey}")
          if key in evals[k]["test_std_per_sample"]:
            if metrickey in evals[k]["test_std_per_sample"][key]:
              data[k,r] = np.nanmean(evals[k]["test_std_per_sample"][key][metrickey])
              r += 1
              if k == 0:
                cs.append(f"val_STD-{key}-{metrickey}")
    idx.append("Mean")
    idx.append("Std")
    data[len(seqs),:] = np.nanmean(data[0:len(seqs),:], axis=0)
    data[len(seqs)+1,:] = np.nanstd(data[0:len(seqs),:], axis=0)
    data = pd.DataFrame(data, columns=cs, index=idx)

    # Plot across folds
    d = data.iloc[0:data.shape[0]-2,:]
    ax = fig.add_subplot(gs[-1,0])
    d1 = d.loc[:,[col for col in d.columns if 'loss' in col]]
    sns.boxplot(data=d1, ax=ax, palette=plt.cm.rainbow(np.linspace(0,1,len([c for c in d.columns if 'loss' in c]))).tolist())
    sns.stripplot(data=d1, jitter=False, palette='dark:black', size=10, alpha=0.5, ax=ax)
    ax.set_title(f"Final losses across folds for {eval_mode} model")

    ax = fig.add_subplot(gs[-1,1:])
    d1 = d.loc[:,[col for col in d.columns if 'loss' not in col]]
    sns.boxplot(data=d1, ax=ax, palette=plt.cm.rainbow(np.linspace(0,1,len([c for c in d.columns if 'loss' not in c]))).tolist())
    sns.stripplot(data=d1, jitter=False, palette='dark:black', size=10, alpha=0.5, ax=ax)
    ax.set_title(f"Final metrics across folds for {eval_mode} model")
    ax.tick_params(axis="x", labelrotation=90)

    plt.tight_layout()
    plt.show()

    pd.set_option('display.max_columns', None)
    display.display(data)

    return data

  def browse_predict(self, seqs, mode="best"):
    """Interactive widget to browse predicted data in notebooks.

    This runs tensorflow in a sub-process for the prediction, so GPU resources can easily be released.That
    also means usually only one prediction browsing process can be active, unless there are sufficient GPU
    resources. To restart a stopped browsing process, restart the jupyter notebook cell.

    Args:
      * `seqs`: A list of keras Sequences as data generator (`bca.dataset.SeqGen`); each element of the list
        is assumed to be a pair of a training and test sequence. All sequences and their results can be viewed.
      * `mode`: "last" or "best" to decide which model to evaluate on (we generally assume "best")
    """
    from ipywidgets import interact, IntSlider, Dropdown, Button

    # Run prediction in separate process (to be able to easily kill tensorflow); communication via queues
    if Cfg.val["multiprocessing"]:
      inp_q = multiprocessing.Queue()
      out_q = multiprocessing.Queue()
      p = multiprocessing.Process(target=Trainer._browse_predict_proc,args=(inp_q,out_q))
      p.start()
    else:
      inp_q = None
      ouot_q = None
      p = None

    # First sample
    cseq = 1
    cset = 0
    cid = int(len(seqs[cseq-1][cset].names)/2)
    X, Y = seqs[cseq-1][cset].cache.get(seqs[cseq-1][cset].names[cid-1])
    if Cfg.val["multiprocessing"]:
      inp_q.put({"path": os.path.join(self._model_path(seqs[cseq-1][0]),mode), "X": X})
      P = out_q.get()
    else:
      P = type(self)._browse_predict(os.path.join(self._model_path(seqs[cseq-1][cset]),mode), X)

    def view(idx, slice, set, seq, overlay):
      # Display set of slices
      nonlocal cid, cset, cseq, inp_q, out_q, X, Y, P, p
      if Cfg.val["multiprocessing"] and p == None:
        return
      set = 0 if set == "train" else 1
      if cid != idx or cset != set or cseq != seq:
        cid = idx
        cset = set
        cseq = seq
        X, Y = seqs[cseq-1][cset].cache.get(seqs[cseq-1][cset].names[cid-1])
        if Cfg.val["multiprocessing"]:
          inp_q.put({"path": os.path.join(self._model_path(seqs[cseq-1][cset]),mode), "X": X})
          P = out_q.get()
        else:
          P = type(self)._browse_predict(os.path.join(self._model_path(seqs[cseq-1][cset]),mode), X)
      x_size = np.sum([x.shape[-1] for x in X])
      y_size = np.sum([y.shape[-1] for y in Y])
      fig, ax = plt.subplots(1,x_size+y_size,sharex=True,sharey=True,dpi=Cfg.val["screen_dpi"],figsize=(Cfg.val["figsize"][0]*(x_size+y_size),Cfg.val["figsize"][1]))
      ax_idx = 0
      for l in range(0,len(X)):
        for k in range(0,X[l].shape[-1]):
          ax[ax_idx].imshow(X[l][:,:,slice-1,k], cmap=Cfg.val["brain_cmap"], interpolation='nearest')
          ax[ax_idx].set_title(seqs[cseq-1][cset].names[cid-1]+"-"+seqs[cseq-1][cset].cache.channels[seqs[cseq-1][cset].cache.inp_chs_idx[l][k]])
          ax_idx += 1
      for l in range(0,len(Y)):
        for k in range(0,Y[l].shape[-1]):
          if "GroundTruth" in overlay:
            ax[ax_idx].imshow(Y[l][:,:,slice-1,k], cmap=Cfg.val["gt_cmap"], interpolation='nearest')
          if "Prediction" in overlay:
            ax[ax_idx].imshow(P[l][0,:,:,slice-1,k], cmap=Cfg.val["pr_cmap"], interpolation='nearest', alpha=0.5)
          ax[ax_idx].set_title(seqs[cseq-1][cset].names[cid-1]+"-"+seqs[cseq-1][cset].cache.channels[seqs[cseq-1][cset].cache.out_chs_idx[l][k]])
          ax_idx += 1
      plt.tight_layout()
      plt.show()

    # Start/stop button
    if Cfg.val["multiprocessing"]:
      btn = Button(description="Stop")
      def btn_clicked(b):
        nonlocal inp_q, p
        inp_q.put({"path": "quit"})
        try:
          p.join()
        except:
          p.kill()
        if p.exitcode != 0:
          raise RuntimeError("Prediction process failed")
        b.description="Prediction Halted"
        p = None
      btn.on_click(btn_clicked)
      display.display(btn)

    # Start interactive widget
    idx_widget = IntSlider(min=1, max=len(seqs[cseq-1][cset].names))
    set_widget = Dropdown(options=["test", "train"], value="train")
    def update_idx_range(*args):
      idx_widget.max = len(seqs[cseq-1][0 if set_widget.value == "train" else 1].names)
    set_widget.observe(update_idx_range, 'value')
    seq_widget = Dropdown(options=[l for l in range(1,len(seqs)+1)], value=cseq)
    overlay_widget = Dropdown(options=["GroundTruth+Prediction", "GroundTruth", "Prediction"], value="GroundTruth+Prediction")
    interact(view, idx=idx_widget, slice=(1,X[0].shape[2]), set=set_widget, seq=seq_widget, overlay=overlay_widget)

  @staticmethod
  def _browse_predict_proc(inp_q, out_q):
    # Prediction in separate process, via queue's, taking in
    # task dictionary with path and X argument.
    path = None
    model = None
    while True:
      task = inp_q.get()
      if task["path"] == "quit":
        break
      # Get model
      if path != task["path"]:
        model = tf.keras.models.load_model(task["path"], custom_objects=Trainer.custom_objects)
        path = task["path"]
      X = task["X"]
      if len(X) > 1:
        P = model.predict([tf.expand_dims(X[l], axis=0) for l in range(0,len(X))], verbose=0)
      else:
        P = model.predict(tf.expand_dims(X[0], axis=0), verbose=0)
      if not isinstance(P,list):
        P = [P]
      out_q.put(P)

  # Static variables for _browse_predict in current process
  _predict_path = None
  _predict_model = None

  @staticmethod
  def _browse_predict(path, X):
    # Prediction in current process
    if path != Trainer._predict_path:
      Trainer._predict_model = tf.keras.models.load_model(path, custom_objects=Trainer.custom_objects)
      Trainer._predict_path = path
    if len(X) > 1:
      P = Trainer._predict_model.predict([tf.expand_dims(X[l], axis=0) for l in range(0,len(X))], verbose=0)
    else:
      P = Trainer._predict_model.predict(tf.expand_dims(X[0], axis=0), verbose=0)
    if not isinstance(P,list):
      P = [P]
    return P

  # Custom objects for loading models
  custom_objects = {
    dsc.__name__: dsc,
    dsc_loss.__name__: dsc_loss,
    iou.__name__: iou,
    WeightedDiceLoss.__name__: WeightedDiceLoss,
    WeightedDiceScore.__name__: WeightedDiceScore,
    sDSC.__name__: sDSC,
    hd95.__name__: hd95,
    sensitivity.__name__: sensitivity,
    specificity.__name__: specificity
  }
