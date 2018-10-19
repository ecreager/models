# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Creates TFRecords of Open Images dataset for object detection.

Example usage:
  python object_detection/dataset_tools/create_oid_tf_record.py \
    --input_box_annotations_csv=/path/to/input/annotations-human-bbox.csv \
    --input_image_label_annotations_csv=/path/to/input/annotations-label.csv \
    --input_images_directory=/path/to/input/image_pixels_directory \
    --input_label_map=/path/to/input/labels_bbox_545.labelmap \
    --output_directoryu/path/to/output/prefix.tfrecord

CSVs with bounding box annotations and image metadata (including the image URLs)
can be downloaded from the Open Images GitHub repository:
https://github.com/openimages/dataset

This script will include every image found in the input_images_directory in the
output TFRecord, even if the image has no corresponding bounding box annotations
in the input_annotations_csv. If input_image_label_annotations_csv is specified,
it will add image-level labels as well. Note that the information of whether a
label is positivelly or negativelly verified is NOT added to tfrecord.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys
from time import time
import threading

import contextlib2
import numpy as np
import pandas as pd
import tensorflow as tf

from object_detection.dataset_tools import oid_tfrecord_creation
from object_detection.dataset_tools import oid_geotagged_tfrecord_creation
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import label_map_util


tf.flags.DEFINE_string(
  'filenames_filename', None,
  'text file containing training set filenames') 
tf.flags.DEFINE_string(
  'input_box_annotations_csv', None,
  'Path to CSV containing image bounding box annotations')
tf.flags.DEFINE_string(
  'input_geotags_csv', None,
  'Path to CSV containing image-level geotags annotations')
tf.flags.DEFINE_string(
  'input_images_directory', None,
  'Directory containing the image pixels '
  'downloaded from the OpenImages GitHub repository.')
tf.flags.DEFINE_string(
  'input_image_label_annotations_csv', None,
  '(optional) Path to CSV containing image-level labels annotations')
tf.flags.DEFINE_string(
  'input_label_map', None, 'Path to the label map proto')
tf.flags.DEFINE_string(
  'input_country_map', None, 'Path to the country map proto')
tf.flags.DEFINE_string(
    'output_directory', None, 
  'Path to the output TFRecord. The shard index and the number of shards '
  'will be appended for each output shard.')
tf.flags.DEFINE_integer(
  'num_shards', 1024, 'Number of TFRecord shards')
tf.app.flags.DEFINE_integer(
  'num_threads',  16, 'Number of threads to preprocess the images.')
tf.flags.DEFINE_string(
  'name', 'train', 
  'unique indentifier specifying the dataset or dataset split')

FLAGS = tf.flags.FLAGS


def _process_image_files_batch(dataframe, thread_index, ranges, name, num_shards, label_map):
  """adapted from models.research.inception.inception.data.build_imagenet_data
  
  TODO: make sure this docstring looks okay
  
  Processes and saves list of images as TFRecord in 1 thread.
    Args:
      dataframe: pd.DataFrame, the dataframe this thread should process
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      num_shards: integer number of shards for this data set.
      label_map: the label map
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  num_files_in_thread = len(dataframe)
  shard_ranges = np.linspace(0, num_files_in_thread, num_shards_per_batch + 1).astype(int)
  
  counter = 0
  
  # split
  sub_dataframes  = [
    dataframe[start:stop] for start, stop in zip(shard_ranges[:-1], shard_ranges[1:])
  ]
  
  # group
  sub_groupbys = [
    sd.groupby('ImageID', sort=False) for sd in sub_dataframes
  ]
  
  # determine whether to include country geotag in records
  if FLAGS.input_geotags_csv:
    get_example = oid_geotagged_tfrecord_creation.tf_example_from_annotations_data_frame
  else:
    get_example = oid_tfrecord_creation.tf_example_from_annotations_data_frame
  
  for s, groupby in enumerate(sub_groupbys):
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    
    writer = tf.python_io.TFRecordWriter(output_file)
    
    shard_counter = 0
    for idx, image_data in enumerate(groupby):
      image_id, image_annotations = image_data
      # In OID image file names are formed by appending ".jpg" to the image ID.
      image_path = os.path.join(FLAGS.input_images_directory, image_id + '.jpg')
      with tf.gfile.Open(image_path, 'rb') as image_file:
        encoded_image = image_file.read()

         
      tf_example = get_example(image_annotations, label_map, encoded_image)
      if tf_example:
        writer.write(tf_example.SerializeToString())
        shard_counter += 1
        counter += 1
        
      if not counter % 100:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread)) 
        sys.stdout.flush()

    # TODO figure out why it writes a single file every time
    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()



def main(_):
  assert not FLAGS.num_shards % FLAGS.num_threads, (
    'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = [
      'input_box_annotations_csv', 'input_images_directory', 'input_label_map',
      'output_directory', 'input_geotags_csv', 'input_country_map'
  ]
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))
      
  if FLAGS.input_geotags_csv:
    assert FLAGS.input_country_map, 'use both or neither'

  label_map = label_map_util.get_label_map_dict(FLAGS.input_label_map)
  if FLAGS.input_country_map:
    label_map = [
      label_map_util.get_label_map_dict(FLAGS.input_label_map),
      label_map_util.get_label_map_dict(FLAGS.input_country_map)
    ]
  else:
      label_map = label_map_util.get_label_map_dict(FLAGS.input_label_map),
  all_box_annotations = pd.read_csv(FLAGS.input_box_annotations_csv)
  if FLAGS.input_image_label_annotations_csv:
    all_label_annotations = pd.read_csv(FLAGS.input_image_label_annotations_csv)
    all_label_annotations.rename(
        columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)
  else:
    all_label_annotations = None
  if FLAGS.filenames_filename and tf.gfile.Exists(FLAGS.filenames_filename):
      print('reading labels from file {}'.format(FLAGS.filenames_filename))
      all_images = tf.gfile.Open(FLAGS.filenames_filename, 'r').readlines()
      all_images = [f.split('\n')[0] for f in all_images]  # annoying
  else: 
    tf.logging.log('globbing')
    all_images = tf.gfile.Glob(
        os.path.join(FLAGS.input_images_directory, '*.jpg'))
  all_image_ids = [os.path.splitext(os.path.basename(v))[0] for v in all_images]
  all_image_ids = pd.DataFrame({'ImageID': all_image_ids})
  if FLAGS.input_geotags_csv:
    all_geotags = pd.read_csv(tf.gfile.Open(FLAGS.input_geotags_csv, 'r'))
  else:
    all_geotags = None
    
      
  all_annotations = pd.concat(
      [all_box_annotations, all_image_ids,
       all_label_annotations, all_geotags])

  tf.logging.log(tf.logging.INFO, 'Found %d images...', len(all_image_ids))

  #with contextlib2.ExitStack() as tf_record_close_stack:
  
  # following code adapted from 
  # adapted from models.research.inception.inception.data.build_imagenet_data
  #
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  num_files = len(all_annotations.ImageID.unique())
  num_labels = len(all_annotations)
  spacing = np.linspace(0, num_labels, FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])


  # sort
  all_annotations = all_annotations.sort_values(by=['ImageID'])

  # split
  annotations_per_thread = [
    all_annotations[r[0]:r[1]] for r in ranges
  ]


  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  threads = []
  for thread_index, annotation_df in enumerate(annotations_per_thread):
    args = (annotation_df, 
            thread_index, ranges, FLAGS.name, 
            FLAGS.num_shards, label_map)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), num_files))
  sys.stdout.flush()


if __name__ == '__main__':
  tf.app.run()
