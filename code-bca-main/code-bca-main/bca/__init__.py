# bca/__init__.py - BCa module
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
# BCa - Brain Cancer Segmentation with Deep Learning

This module provides functions and classes in the following sub-modules to handle 
brain cancer datasets and train deep tensorflow models.

## General Modules

Modules with general functionality for datasets and models:
  * **bca.dataset**: Represent brain cancer datasets on disk and create keras sequences for deep learning.
  * **bca.trainer**: Train models using the dataset sequences with tensorflow.
  * **bca.model**: Base class for model generator classes (used by models below).
  * **bca.loss**: Custom loss classes.
  * **bca.metric**: Custom metrics classes.
  * **bca.interpret**: Explainability functionality.
  * **bca.scheduler**: Scheduler to train models remotely from jupyter notebooks or from the command line
    (mostly limited to Linux as `rsync` and `ssh` are needed). Tasks for the scheduler are created by trainer.

## Models

The following deep learning models are available to use:

  * **bca.unet**: 3D Unet architecture.
  * **bca.latupnet**: LATUP-Net architecture.

## Misc

Other sub-modules are:
  * **bca.cfg**: Provides a configuration class to hold the BCa configuration. This is mostly for internal use, but
    also provides configuration options.
"""

__all__ = ["dataset", "model", "trainer", "scheduler", "unet", "latupnet", "loss", "metric", "interpret", "cfg"]
