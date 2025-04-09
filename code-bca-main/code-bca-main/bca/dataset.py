# bca/dataset.py - Brain cancer dataset
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Ebtihal Alwadee <AlwadeeEJ@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

from .cfg import Cfg

import os
import sys
import warnings
import psutil
import joblib
import csv
import random
import numpy as np
import nibabel as nib
import nibabel.processing as nibp

from tensorflow.keras.utils import Sequence

class Dataset:
  """Represents original dataset to generate training sequences from, with preprocessing.

  This is an iterator of the dataset and also implements `len` and `[idx]` operators and
  can be printed and converted to a string.
  """

  def __init__(self, folder, cache, seg_mask="seg"):
    """Create a dataset from folder following BraTS structure.

    Args:
      * `folder`: Folder containing original dataset;
      * `cache`: Cache folder for pre-processed data;
      * `seg_mask`: Name of segmentation mask channel.
    """
    self.folder = folder
    self.cache = cache
    self.seg_mask = seg_mask

    # Find all patient folders and channels available for all of them
    if Cfg.log(Cfg.Info):
      print(f"# Initialising dataset {self.folder}")
    self.patients = []
    self.channels = []
    for f in sorted(os.listdir(self.folder)):
      p_fldr = os.path.join(self.folder,f)
      p_chs = []
      if os.path.isdir(p_fldr) and f[0] != '.':
        try:
          self.patients.append(f)
          id = int(f.split("_")[-1])
          for ff in os.listdir(p_fldr):
            if ff[-4:] == ".nii" or ff[-7:] == ".nii.gz":
              p_chs.append(ff.split(".")[0].split("_")[-1].lower())
          if len(self.channels) == 0:
            self.channels = p_chs
          else:
            for c in self.channels:
              if c not in p_chs:
                self.channels.remove(c)
        except:
          pass
    self.channels.sort()
    if Cfg.log(Cfg.Info):
      print(f"  Patients: {len(self)}")
      print(f"  Channels: "+", ".join(self.channels))

    if len(self) < 1:
      warnings.warn("No patients found")

    # Crop
    self.crops = None
    self.crops_type = "orig"

  def filter_low_labels(self, c, min_label_per):
    """Remove samples with small number of labels in channel `c` from dataset.

    Args:
      * `c`: Channel name used to make decision on (must be a segmentation mask);
      * `min_label_per`: minimum percentage of data in sample.
    """
    # Cache for voxel counts
    cache_data = []
    cache_updated = False
    if self.crops_type == "f":
      data_cache = "f"+"x".join([str(c[0])+"_"+str(c[1]) for c in self.crops])
    elif self.crops_type == "bb" or self.crops_type == "orig":
      data_cache = self.crops_type
    else:
      raise RuntimeError(f"Unknown crop type {self.crops_type}")
    cache = os.path.join(self.cache, f"voxel_counts_{data_cache}.csv")
    voxel_labels = [None] * len(self)
    voxel_counts = [None] * len(self)
    if os.path.isfile(cache):
      with open(cache, "r") as f:
        rows = csv.reader(f)
        for row in rows:
          try:
            idx = self.patients.index(row[0])
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
    # Update voxel counts
    remove_idx = []
    for k in reversed(range(0,len(self))): # Reversed to delete from end to start, so indices remain valid
      if voxel_labels[k] is not None and voxel_counts[k] is not None:
        labels = voxel_labels[k]
        label_counts = voxel_counts[k]
      else:
        fn = os.path.join(self.folder, self.patients[k], self.patients[k]+'_'+c+'.nii')
        if not os.path.isfile(fn):
          fn += ".gz"
        data = nib.load(fn)
        if self.crops_type != "orig":
          # Respect crop for percentage
          if self.crops_type == "f":
            crp = self.crops
          elif self.crops_type == "bb":
            crp = self.crops[k]
          else:
            raise RuntimeError(f"Illegal crops {self.crops_type}")
          data = data.slicer[crp[1][0]:crp[1][1],crp[0][0]:crp[0][1],crp[2][0]:crp[2][1]]
        data = data.get_fdata()
        labels, label_counts = np.unique(data, return_counts=True)
        labels = [int(l) for l in labels]
        voxel_labels[k] = labels
        voxel_counts[k] = label_counts
        cache_updated = True
      cnt_a = np.sum(label_counts)
      cnt_l = np.sum([label_counts[l] for l in range(0,len(labels)) if labels[l] != 0])
      if cnt_l / cnt_a < min_label_per:
        if Cfg.log(Cfg.Info):
          print(f"Drop {self.patients[k]} - {cnt_l / cnt_a}%")
        remove_idx.append(k)
    # Update cache
    if cache_updated:
      os.makedirs(self.cache, exist_ok=True)
      if os.path.exists(cache):
        os.remove(cache)
      cache_data = [None] * len(self)
      with open(cache, "w") as f:
        out = csv.writer(f)
        for idx in range(0,len(self)):
          out.writerow([self.patients[idx]] + [str(voxel_labels[idx][col])+":"+str(voxel_counts[idx][col]) for col in range(0,len(voxel_labels[idx]))])
    # Delete indices
    for l in remove_idx:
      del self.patients[l]
      if self.crops_type == "bb":
        del self.crops[l]

  def __repr__(self):
    return f"bca.dataset.Dataset({self.folder})"

  def __str__(self):
    str = f"# Dataset: {self.folder} [{len(self)} patients]\n" + \
          f"Cache: {self.cache}\n"
    if len(self) > 0:
      str += f"Channels (patient {self.patients[0]}):\n"
      data = self[0]
      for c in self.channels:
        str += f"  {c}: {data[c].shape} ({data[c].get_data_dtype()})\n"
    else:
      str += "Empty dataset (no patients found)"
    if self.crops_type != "orig":
      if self.crops_type == "f":
        str += f"Crop: {self.crops}\n"
      elif self.crops_type == "bb":
        str += "Crop: bounding-box\n"
      else:
        str += f"Crop: UNKNOWN {self.crops_type}\n"
    return str

  def __iter__(self):
    # Iterator via patients dictionary: init
    self._iter = iter(self.patients)
    return self

  def __next__(self):
    # Iterator via patients dictionary: next
    return next(self._iter)

  def __len__(self):
    # Number of patients for len(.)
    return len(self.patients)

  def __getitem__(self, idx):
    # Load nii common-modalities from patient array index
    data = {}
    for c in self.channels:
      fn = os.path.join(self.folder, self.patients[idx], self.patients[idx]+'_'+c+'.nii')
      if not os.path.isfile(fn):
        fn += ".gz"
      data[c] = nib.load(fn)
    return data

  def cropped(self, idx, channels=None):
    """Load nii common-modalities from patient array index and apply crop.

    Args:
      * `idx`: patient/sample index in dataset.
      * `channel`: if not none, only load specified channels; otherwise load all specified in the dataset object.
    
    Return:
      * Cropped patient data.
    """
    data = {}
    if channels is None:
      channels = self.channels
    for c in channels:
      fn = os.path.join(self.folder, self.patients[idx], self.patients[idx]+'_'+c+'.nii')
      if not os.path.isfile(fn):
        fn += ".gz"
      data[c] = nib.load(fn)
      if self.crops_type != "orig":
        # Respect crop for percentage
        if self.crops_type == "f":
          crp = self.crops
        elif self.crops_type == "bb":
          crp = self.crops[idx]
        else:
          raise RuntimeError(f"Illegal crops {self.crops_type}")
        data[c] = data[c].slicer[crp[0][0]:crp[0][1],crp[1][0]:crp[1][1],crp[2][0]:crp[2][1]]
    return data

  def patient_name(self,idx):
    """Map patient index to (folder) name.

    Args:
      * `idx`: patient/sample index in dataset.
    
    Return:
      * Patient folder name.
    """
    return self.patients[idx]

  def patient_idx(self,pid):
    """Find index for patient pid.

    Args:
      * `pid`: patient/sample name.

    Return:
      * Patient index in dataset.
    """
    return self.patients.index(pid)

  def browse(self):
    """Interactive widget to browse data in notebooks.
    """
    if len(self) < 1:
      raise RuntimeError("Dataset empty")
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as patches
    from ipywidgets import interact
    # Custom color map for BraTS labels
    seg_cmap = ListedColormap([Cfg.val["col_label0"],
                               Cfg.val["col_label1"],
                               Cfg.val["col_label2"],
                               Cfg.val["col_label3"],
                               Cfg.val["col_label4"]])
    seg_norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5,4.5], seg_cmap.N)
    cid = 1
    data = self[cid-1]
    def view(idx, slice, overlay):
      # Display set of slices for patient id
      nonlocal cid, data
      if cid != idx:
        # Caching data
        cid = idx
        data = self[cid-1]
      fig, ax = plt.subplots(1,len(self.channels),sharex=True,sharey=True,dpi=Cfg.val["screen_dpi"],figsize=(Cfg.val["figsize"][0]*len(self.channels),Cfg.val["figsize"][1]))
      for k, c in enumerate(self.channels):
        stack = data[c].get_fdata()
        if c == self.seg_mask:
          ax[k].imshow(stack[:,:,slice-1], cmap=seg_cmap, norm=seg_norm, interpolation='nearest')
        else:
          ax[k].imshow(stack[:,:,slice-1], cmap=Cfg.val["brain_cmap"], interpolation='nearest')
        if overlay == self.seg_mask and c != self.seg_mask:
          # Overlay segmentation
          stack = data[self.seg_mask].get_fdata()
          ax[k].imshow(stack[:,:,slice-1], cmap=seg_cmap, norm=seg_norm, interpolation='nearest', alpha=0.5)
        if c == self.seg_mask: 
          # Labels for segmentation mask
          vals = [int(v) for v in np.unique(stack[:,:,slice-1]) if v > 0.0]
          if len(vals) > 0:
            pats = [patches.Patch(color=seg_cmap(v), label=str(v)) for v in vals]
            ax[k].legend(handles=pats, loc=0, borderaxespad=0.1)
        if self.crops is not None:
          # Indicate crop
          if self.crops_type == "f":
            if slice-1 >= self.crops[2][0] and slice-1 <= self.crops[2][1]:
              rect = patches.Rectangle((self.crops[0][0]-1,self.crops[1][0]-1),
                                       self.crops[0][1] - self.crops[0][0]+1,
                                       self.crops[1][1] - self.crops[1][0]+1, 
                                       linewidth=1, edgecolor='r', facecolor='none')
              ax[k].add_patch(rect)
          elif self.crops_type == "bb":
            if slice-1 >= self.crops[idx-1][2][0] and slice-1 <= self.crops[idx-1][2][1]:
              rect = patches.Rectangle((self.crops[idx-1][0][0]-1,self.crops[idx-1][1][0]-1),
                                       self.crops[idx-1][0][1] - self.crops[idx-1][0][0]+1,
                                       self.crops[idx-1][1][1] - self.crops[idx-1][1][0]+1, 
                                       linewidth=1, edgecolor='r', facecolor='none')
              ax[k].add_patch(rect)
          else:
            raise RuntimeError(f"Illegal crops {self.crops_type}")
        ax[k].set_title(self.patients[idx-1]+"-"+c)
      plt.tight_layout()
      plt.show()
    # Start interactive widget
    interact(view, idx=(1,len(self)), slice=(1,data[self.channels[0]].shape[-1]), overlay=["None", self.seg_mask])

  def crop(self, xr, yr, zr):
    """Set single crop region.

    Args:
      * `xr`, `yr`, `zr`: tuples of index ranges indicating crop.
    """
    self.crops_type = "f"
    self.crops = [xr,yr,zr]

  def crop_to_bb(self):
    """Crop to bounding box per patient across all channels.

    Computes a custom crop based on the empty region across all channels per sample/patient.
    """
    # Load from cache, if valid
    self.crops_type = "bb"
    self.crops = [None] * len(self)
    cache_data = []
    cache_updated = False
    cache = os.path.join(self.cache, "crop_bb.csv")
    if os.path.isfile(cache):
      with open(cache, "r") as f:
        rows = csv.reader(f)
        for row in rows:
          try:
            idx = self.patients.index(row[0])
            self.crops[idx] = [[int(row[1]),int(row[2])],[int(row[3]),int(row[4])],[int(row[5]),int(row[6])]]
          except ValueError:
            pass
          cache_data.append(row)
    # Find missing bounding boxes
    indices = [k for k in range(0,len(self)) if self.crops[k] is None]
    if len(indices) > 0:
      new_crops = joblib.Parallel(n_jobs=Cfg.val['parallel'], prefer="threads")(joblib.delayed(Dataset._bb_stack)(self[k]) for k in indices)
      for k in range(0,len(indices)):
        self.crops[indices[k]] = new_crops[k]
        cache_data.append([self.patient_name(indices[k]), *np.array(new_crops[k]).flatten()])
        cache_updated = True
    # Cache results
    if cache_updated:
      os.makedirs(self.cache, exist_ok=True)
      if os.path.exists(cache):
        os.remove(cache)
      cache_data.sort(key=lambda x : x[0])
      with open(cache, "w") as f:
        out = csv.writer(f)
        for k in range(0,len(cache_data)):
          out.writerow(cache_data[k])

  @staticmethod
  def _bb_stack(data):
    # Find bounding box of 3D stack with index k
    indices = np.where(np.sum([np.abs(data[c].get_fdata()) for c in data], axis=0) > 0)
    return  [[np.min(indices[1]),np.max(indices[1])], # Note, row/column vs width/height!
             [np.min(indices[0]),np.max(indices[0])],
             [np.min(indices[2]),np.max(indices[2])]]

  def sequences(self, k, dim, inp, out, batch_size, pre_proc=None, seed=None, fixed_batch_size=False):
    """Create sequence generators for train,test pairs of all sets.

    Args:
      * `k`: split indicator:
        * `]0,1[`: percentage of data in training set;
        * `1`: all data goes to training set;
        * `>1`: k-fold cross-validation split
      * `dim`: Spatial dimension of data to rescale it to;
      * `inp`: List or list of lists of input channel names (for single or multiple inputs);
      * `out`: List or list of lists of output channel names (for single or multiple outputs);
        * For both a special notation can be used for the segmentation mask:
          * "seg+1+2+3" combines labels 1, 2, 3 to a single 0,1 mask;
          * "seg=1=2" keeps labels 1,2 as is in the channel and removes any other labels;
      * `batch_size`: Batch size for processing;
      * `pre_proc`: Function to use for pre-processing data (see `norm_minmax` and `norm_histeq_mask` below),
        `None` for no pre-processing;
      * `seed`: Random number generator seed for split;
      * `fixed_batch_size`: Boolean indicating whether sequence generators should produce a fixed sized batches.
    """
    # Init seed/rng for split reproducibility
    if seed is None:
      seed = random.randrange(sys.maxsize)

    # Split
    if k >= 1:
      k = int(k)
    self._split(k, seed)

    # Cache name
    if self.crops_type == "f":
      data_cache = "f"+"x".join([str(c[0])+"_"+str(c[1]) for c in self.crops])
    elif self.crops_type == "bb" or self.crops_type == "orig":
      data_cache = self.crops_type
    else:
      raise RuntimeError(f"Illegal crops {self.crops_type}")
    data_cache = os.path.join(self.cache, data_cache+"-"+"_".join([str(d) for d in dim])+"-"+("none" if pre_proc is None  else pre_proc.__name__))
    os.makedirs(data_cache, exist_ok=True)

    # Create samples in cache
    joblib.Parallel(n_jobs=Cfg.val['parallel'], prefer="threads")(joblib.delayed(self._create_sample)(p, dim, pre_proc, data_cache, self.seg_mask) for p in range(0,len(self)))

    # Memory cache across sequences
    cache = Cache(data_cache, self.channels, dim, inp, out, self.seg_mask)

    # Create train/test pair sequences
    if k == 1:
      # Sequence for single set, no train/test split
      ns = [self.patients[m] for m in range(0,len(self))]
      seqs = [(SeqGen(ns, cache, dim, batch_size, k=1, k_n=0, seed=0, shuffle=True, fixed_batch_size=fixed_batch_size), None)]
    elif k <= 1:
      # Train set is labelled 0, as only one split
      ns_train = [self.patients[m] for m in range(0,len(self)) if self.set[m] == 0]
      ns_test = [self.patients[m] for m in range(0,len(self)) if self.set[m] != 0]
      seqs = [(SeqGen(ns_train, cache, dim, batch_size, k=k, k_n=0, seed=seed, shuffle=True, fixed_batch_size=fixed_batch_size),
               SeqGen(ns_test,  cache, dim, batch_size, k=k, k_n=0, seed=seed, shuffle=False, fixed_batch_size=fixed_batch_size))]
    else:
      seqs = []
      # Train set is labelled != l, as multiple folds
      for l in range(0,k):
        # Sequence for fold l
        ns_train = [self.patients[m] for m in range(0,len(self)) if self.set[m] != l]
        ns_test = [self.patients[m] for m in range(0,len(self)) if self.set[m] == l]
        seqs.append((SeqGen(ns_train, cache, dim, batch_size, k=k, k_n=l, seed=seed, shuffle=True, fixed_batch_size=fixed_batch_size),
                     SeqGen(ns_test,  cache, dim, batch_size, k=k, k_n=l, seed=seed, shuffle=False, fixed_batch_size=fixed_batch_size)))
    return seqs

  def _split(self, k, seed):
    # Split dataset for training
    #   k = 0: one set
    #   k \in (0,1): k% split
    #   k in 1,2,3...: k-fold split
    if k == 1:
      # Single dataset
      if  Cfg.log(Cfg.Info):
        print(f"# Single dataset (no split): {len(self)} set")
      self.set = np.zeros(len(self),dtype=np.uint8)
    elif k > 0 and k < 1:
      # two-fold split
      if Cfg.log(Cfg.Info):
        print(f"#  {k*100} : {100 - k*100} % split: ({int(k*len(self))},{len(self)-int(k*len(self))}) sets")
      # Simple two-fold split, with shuffle
      idx = np.arange(0,len(self))
      np.random.default_rng(seed=seed).shuffle(idx)
      split = np.round(len(self) * k).astype(np.uint64)
      self.set = np.ndarray(len(self),dtype=np.uint8)
      self.set[idx[0:split]] = 0
      self.set[idx[split:]] = 1
    elif int(k) > 1:
      # k-fold split
      k = int(k)
      if Cfg.log(Cfg.Info):
        print(f"# {k}-fold split of {len(self)}: {len(self) // k} per set; {k-len(self) % k} set(s) with one more")
      # Venetian blinds k-fold split with shuffle
      idx = np.arange(0,len(self))
      np.random.default_rng(seed=seed).shuffle(idx)
      self.set = np.floor(idx % k).astype(np.uint8)
    else:
      raise RuntimeError(f"Illegal k: {k}")

  def _create_sample(self, pidx, dim, pre_proc, data_cache, seg_mask):
    # Create single input/output sample in cache
    if os.path.isfile(os.path.join(data_cache,self.patient_name(pidx)+".npy")):
      return
    # Setup data array
    data = self[pidx]
    label_counts = None
    unique_labels = None
    X = np.empty((*dim, len(self.channels)), dtype=np.float32)
    for s,c in enumerate(self.channels):
      # Crop and determine scale
      if self.crops_type == "orig":
        stack = data[c]
        vs = stack.header.get_zooms()
        sx = vs[0]
        sy = vs[1]
        sz = vs[2]
      else:
        if self.crops_type == "f":
          crp = self.crops
        elif self.crops_type == "bb":
          crp = self.crops[pidx]
        else:
          raise RuntimeError(f"Illegal crops {self.crops_type}")
        stack = data[c].slicer[crp[1][0]:crp[1][1],crp[0][0]:crp[0][1],crp[2][0]:crp[2][1]]
        vs = stack.header.get_zooms()
        sx = (crp[1][1] - crp[1][0]) * vs[0] / dim[0]
        sy = (crp[0][1] - crp[0][0]) * vs[1] / dim[1]
        sz = (crp[2][1] - crp[2][0]) * vs[2] / dim[2]
      # Resample 
      if c == seg_mask:
        # Transform mask
        org_data = stack.get_fdata()
        labels = np.unique(org_data)
        masks = np.zeros((len(labels),*dim))
        # Split masks into labels, transform, and combine again
        for l, ul in enumerate(labels):
          # Transform label mask
          new_data = np.array(org_data==ul, dtype=np.float32)
          new_stack = nib.Nifti1Image(new_data, stack.affine, stack.header)
          new_stack = nibp.conform(new_stack, out_shape=dim, voxel_size=(sx,sy,sz), orientation="LPS")
          masks[l,...] = new_stack.get_fdata()
        # Vote for maximum value per voxel
        mask = np.argmax(masks,axis=0)
        # Map indices to labels
        for l in reversed(range(0,len(labels))):
          X[(mask==l),s] = labels[l]
      else:
        # Transform image
        stack = nibp.conform(stack, out_shape=dim, voxel_size=(sx,sy,sz), orientation="LPS")
        X[...,s] = stack.get_fdata().astype(np.float32)
      if pre_proc is not None:
        pre_proc(X[...,s], c, seg_mask)
    np.save(os.path.join(data_cache,self.patient_name(pidx)+".npy"), X)

  @staticmethod
  def norm_minmax(x, ch, seg_mask):
    """Function to normalise input and output samples mapping `[min,max]` to `[0,1] in place.

    This can be a `pre_proc` function for `sequences`.

    Args:
      * `x`: data sample;
      * `ch`: channel name;
      * `seg_mask`: segmentation mask name.
    """
    if ch != seg_mask:
      x -= np.min(x) # Mask background
      x /= np.max(x)

  @staticmethod
  def norm_histeq_mask(x, ch, seg_mask):
    """Function to normalise input and output samples applying histogram normalisation and then mapping the result to `[0,1]` in place.

    This can be a `pre_proc` function for `sequences`.

    Args:
      * `x`: data sample;
      * `ch`: channel name;
      * `seg_mask`: segmentation mask name.
    """
    if ch != seg_mask:
      xf = x.flatten()
      # Mask background
      mask = xf > 0.0
      xf[np.logical_not(mask)] = 0.0 # Background to mask
      # Normalise to have the bins right
      xf[mask] /= np.max(xf[mask])
      # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
      hg, bins = np.histogram(xf[mask], np.linspace(0,1,2**12), density=True)
      # Equalise
      cdf = hg.cumsum()
      cdf /= cdf[-1]
      # Linear interpolation of cdf to find new values
      xf[mask] = np.interp(xf[mask], bins[:-1], cdf)
      np.copyto(x, xf.reshape(x.shape))

class Cache():
  """Cache to store samples for a pre-processed dataset in memory for faster access.

  It stores the samples in memory until we run out (indicating by a config. parameter).
  Then it keeps loading them from disk instead. This is used across a set of sequences
  generated for the same dataset.
  """

  def __init__(self, data_folder, channels, dim, inp_chs, out_chs, seg_mask):
    """Cache of npy loaded from data_folder.

    Args:
      * `data_folder`: Folder to load data from (pre-processed data from Dataset);
      * `channels`: Channel names in sequence of channel index for samples in `data_folder`;
      * `dim`: Spatial dimension of samples in `data_folder`;
      * `inp_chs`: list of lists of input channel names for each input;
      * `out_chs`: list of lists of output channel names for each output;
      * `seg_mask`: segmentation mask channel name.
    """
    self.data_folder = data_folder
    self.channels = channels
    self.dim = dim
    self.seg_mask = seg_mask
    if isinstance(inp_chs[0], list):
      self.inp_chs = inp_chs
    else:
      self.inp_chs = [inp_chs]
    self.inp_chs_idx = [[self._channel_idx(c) for c in ic] for ic in self.inp_chs]
    self.inp_chs_mask = [[self._channel_mask(c) for c in ic] for ic in self.inp_chs]
    if isinstance(out_chs[0], list):
      self.out_chs = out_chs
    else:
      self.out_chs = [out_chs]
    self.out_chs_idx = [[self._channel_idx(c) for c in oc] for oc in self.out_chs]
    self.out_chs_mask = [[self._channel_mask(c) for c in oc] for oc in self.out_chs]
    self.clear()

  def _channel_idx(self, c):
    # Get index in data array of channel to copy
    try:
      return self.channels.index(c)
    except:
      return self.channels.index(self.seg_mask)

  def _channel_mask(self, c):
    # Get processing for mask channel
    if c[0:len(self.seg_mask)] == self.seg_mask:
      if c[len(self.seg_mask)] == "=":
        # Preserve labels listed (use negative index to indicate this for later processing)
        # Preserve =0 as -inf (this works later in the mask generation as 0 would remain 0 anyway)
        return [-np.inf if int(l) == 0 else -int(l) for l in c[(len(self.seg_mask)+1):].split("=")]
      elif c[3] == "+":
        # Combine listed labels to binary mask
        return [int(l) for l in c[(len(self.seg_mask)+1):].split("+")]
    return []

  def clear(self):
    """Clear cache.
    """
    self.cacheX = {}
    self.cacheY = {}
    self.warn = False

  def get(self, id):
    """Get sample.

    Args:
      * `id`: Name of sample to return.
    
    Return:
      * `X`, `Y`: Numpy array for input and output.
    """
    X = [np.ndarray((*self.dim,len(ic)), dtype=np.float32) for ic in self.inp_chs]
    Y = [np.ndarray((*self.dim,len(oc)), dtype=np.float32) for oc in self.out_chs]
    self.copy_to(id,X,Y)
    return X, Y

  def copy_to(self, id, X, Y):
    """Copy sample to memory.

    Args:
      * `id`: Name of sample;
      * `X`: Numpy array to store input;
      * `Y`: Numpy array to store output.
    """
    if id not in self.cacheX:
      # Load
      data = np.load(os.path.join(self.data_folder,id+".npy"))
      # Create X,Y input/output pair
      for V in [(X,self.inp_chs_idx,self.inp_chs_mask), (Y,self.out_chs_idx,self.out_chs_mask)]:
        for k, ci in enumerate(V[1]):
          for l, p in enumerate(ci):
            np.copyto(V[0][k][...,l], data[...,p], casting="no")
            if len(V[2][k][l]) > 0:
              # Process mask - create mask from mask index using copied channel from inp index
              if V[2][k][l][0] >= 0:
                # Combine labels
                midx = (V[0][k][...,l] == V[2][k][l][0])
                for kk in range (1,len(V[2][k][l])):
                  midx |= (V[0][k][...,l] == V[2][k][l][kk])
                V[0][k][...,l] = midx
              else:
                # Preserve labels
                # Note, we represent =0 as -np.inf in the out index; 
                # The works as 0 would be preserved as 0 in any case.
                midx = (V[0][k][...,l] != -V[2][k][l][0])
                for kk in range (1,len(V[2][k][l])):
                  midx &= (V[0][k][...,l] != -V[2][k][l][kk])
                V[0][k][midx,l] = 0
      # Cache
      if    (not self.warn and psutil.virtual_memory().percent < Cfg.val["low_mem_percentage"]) \
         or (    self.warn and psutil.virtual_memory().percent < Cfg.val["low_mem_percentage"]-Cfg.val["low_mem_percentage_restart"]):
        if self.warn:
          print(f"***Warning: memory {psutil.virtual_memory().percent}% full; restarting cache***")
          self.warn = False
        self.cacheX[id] = [np.copy(X[k]) for k in range(0,len(self.inp_chs_idx))]
        self.cacheY[id] = [np.copy(Y[k]) for k in range(0,len(self.out_chs_idx))]
      elif not self.warn:
        print(f"***Warning: memory {psutil.virtual_memory().percent}% full; stopping cache***")
        self.warn = True
    else:
      # Get from cache
      for k in range(0,len(self.cacheX[id])):
        np.copyto(X[k], self.cacheX[id][k])
      for k in range(0,len(self.cacheY[id])):
        np.copyto(Y[k], self.cacheY[id][k])

class SeqGen(Sequence):
  """Keras a sequence of data generator.

  This is used to generate a data sequence for training and testing. It only stores
  the names of the samples and uses `bca.dataset.Cache` to load and cache, as much as
  possible, the samples. It is generates by `bca.dataset.Dataset.sequences`.

  This implements `len` and `[idx]` operators.
  """

  def __init__(self, names, cache, dim, batch_size, k, k_n, seed, shuffle=True, fixed_batch_size=False):
    """Construct a keras data Sequence.

    Args:
      * `names`: List of sample names in the sequence;
      * `cache`: Cache to use to load and cache the data;
      * `dim`: Dimensions of each simple;
      * `batch_size`: Training batch size;
      * `k`: Parameter indicating split of dataset used to generate this sequence;
      * `k_n`: Parameter indicating which fold this sequence belongs to;
      * `seed`: Seed used to generate the split
      * `shuffle`: Boolean indicating whether to shuffle data sequence after each epoch;
      * `fixed_batch_size`: Boolean indicating whether to use fixed batch size; this means batches are
        truncated if they are smaller than the batch size specified.
    """
    super(SeqGen, self).__init__()
    self.names = names
    self.cache = cache
    self.dim = dim
    self.batch_size = batch_size
    self.k = k
    self.k_n = k_n
    self.seed = seed
    self.shuffle = shuffle
    self.idx = np.arange(0,len(self.names))
    self.rng = np.random.default_rng()
    self.fixed_batch_size = fixed_batch_size
    self.on_epoch_end() # Run shuffle on first, too

  def __len__(self):
    # Number of batches per epoch
    batches = len(self.names) // self.batch_size
    # We fix the batch size on request. E.g. if we setup the module with a fixed batch_size for all inputs.
    # This is especially necessary on some systems to avoid errors in some cases, so we also allow to overwrite this with a dev flag.
    # The particular issue we ran into was a count=0 tensor at the bottle neck of a standard UNet3D on a V100-16GB with CUDA 11.5; cudnn 8.{3,6}.
    # A cudnn error was triggered on a count=0 BatchDescriptor. Could not reproduce this on more recent hardware / software.
    if not (self.fixed_batch_size or Cfg.dev("FIXED_BATCH_SIZE")) and batches * self.batch_size < len(self.names):
      batches += 1
    return batches

  def __getitem__(self, index):
    # Generate one batch
    batch_idx = self.idx[index*self.batch_size:min((index+1)*self.batch_size,len(self.names))]
    # Data
    X = [np.empty((len(batch_idx), *self.dim, len(ic)), dtype=np.float32) for ic in self.cache.inp_chs]
    Y = [np.empty((len(batch_idx), *self.dim, len(oc)), dtype=np.float32) for oc in self.cache.out_chs]
    # Load
    for l in range(0,len(batch_idx)):
      self.cache.copy_to(self.names[batch_idx[l]],
                         [X[k][l,:] for k in range(0,len(X))],
                         [Y[k][l,:] for k in range(0,len(Y))])
    if len(X) == 1:
      X = X[0]
    if len(Y) == 1:
      Y = Y[0]
    return X, Y

  def on_epoch_end(self):
    """Method called at the end of every epoch.

    Shuffle indices, if requested.
    """
    if self.shuffle == True:
      self.rng.shuffle(self.idx)

  def disable_shuffle(self):
    """Disable shuffling of data sequence.
    """
    self.shuffle = False
    self.idx = np.arange(0,len(self.names))
  
  def enable_shuffle(self):
    """Enable shuffling of data sequence.
    """
    self.shuffle = True
    self.idx = np.arange(0,len(self.names))
    self.on_epoch_end()

  def clear_cache(self):
    """Clear Cache.
    """
    self.cache.clear()

  def browse(self):
    """Interactive widget to browse data in notebooks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as patches
    from ipywidgets import interact
    cid = 1
    X, Y = self.cache.get(self.names[cid-1])
    seg_cmap = ListedColormap([Cfg.val["col_label0"],
                               Cfg.val["col_label1"],
                               Cfg.val["col_label2"],
                               Cfg.val["col_label3"],
                               Cfg.val["col_label4"]])
    seg_norm = BoundaryNorm([-.5,.5,1.5,2.5,3.5,4.5], seg_cmap.N)
    def view(idx, slice):
      # Display set of slices for patient id
      nonlocal cid, X, Y
      if cid != idx:
        cid = idx
        X, Y = self.cache.get(self.names[cid-1])
      x_size = np.sum([x.shape[-1] for x in X])
      y_size = np.sum([y.shape[-1] for y in Y])
      fig, ax = plt.subplots(1,x_size+y_size,sharex=True,sharey=True,dpi=Cfg.val["screen_dpi"],figsize=(Cfg.val["figsize"][0]*(x_size+y_size),Cfg.val["figsize"][1]))
      ax_idx = 0
      for V in [(X,self.cache.inp_chs_mask),(Y,self.cache.out_chs_mask)]:
        for l in range(0,len(V[0])):
          for k in range(0,V[0][l].shape[-1]):
            if len(V[1][l][k]) > 0:
              ax[ax_idx].imshow(V[0][l][:,:,slice-1,k], cmap=seg_cmap, norm=seg_norm, interpolation='nearest')
              vals = [int(v) for v in np.unique(V[0][l][:,:,slice-1,k]) if v > 0.0]
              if len(vals) > 0:
                pats = [patches.Patch(color=seg_cmap(v), label=str(v)) for v in vals]
                ax[ax_idx].legend(handles=pats, loc=0, borderaxespad=0.1)
              ax[ax_idx].set_title(self.names[cid-1]+"-Label"+str(V[1][l][k]))
            else:
              ax[ax_idx].imshow(V[0][l][:,:,slice-1,k], cmap=Cfg.val["brain_cmap"], interpolation='nearest')
              ax[ax_idx].set_title(self.names[cid-1]+"-"+str(self.cache.channels[self.cache.inp_chs_idx[l][k]]))
            ax_idx += 1
      plt.tight_layout()
      plt.show()
    # Start interactive widget
    interact(view, idx=(1,len(self.names)), slice=(1,X[0].shape[2]))
