# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the PET dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

# TODO(nsilberman): Add tfrecord file type once the script is updated.
_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'Train': 7393,
  	'validation': 1000
}

_ITEMS_TO_DESCRIPTIONS = {
    'image/height': 'the height of the image',
    'image/width': 'the width of the image',
    'image/colorspace': 'one of the RGB or BGR',
    'image/channels': '3',
    'image/class/label': 'the label index of the image',
    'image/class/text': 'the text label of  the image',
    'image/format': 'the format of image',
    'image/filename':  'the filename of image',
    'image/encoded': 'A color image of varying height and width.',
}

_NUM_CLASSES = 37

# If set to false, will not try to set label_to_names in dataset
# by reading them from labels.txt or github.
LOAD_READABLE_NAMES = True

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/height': tf.VarLenFeature(dtype=tf.int64),
      'image/width': tf.VarLenFeature(dtype=tf.int64),
#      'image/colorspace': _bytes_feature(colorspace),
#      'image/channels': _int64_feature(channels),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
      'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
      'filename': slim.tfexample_decoder.Tensor('image/filename')
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
      labels_to_names = dataset_utils.read_label_file(dataset_dir)
  # necessary?
  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
