from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
from absl import app
from absl import flags
from absl import logging

import json
import time
from pathlib import Path

from nasbench.lib import config as nasbench_config
from nasbench.lib.evaluate import _TrainAndEvaluator
from nasbench.lib import model_spec
from nasbench.lib import cifar
from nasbench.lib import model_builder
from nasbench import api as nasbench_api
from nasbench.scripts.run_evaluation import NumpyEncoder

import numpy as np
import tensorflow as tf

from utils import prepare_kd_dataset


# Clear some flags:
for name in [
    'train_data_files', 'valid_data_file',
    'test_data_file', 'sample_data_file',
    'batch_size', 'train_epochs']:
        delattr(flags.FLAGS, name)
        
flags.DEFINE_string('path_to_nasbench', '../data/nasbench_only108.tfrecord',
                    'Path to nasbanch dataset tfrecord file')
flags.DEFINE_string('hash_key', '02e5a0247bbdcf2860b7e96f74961594',
                    'Hash key of nasbench architecture for traning')
flags.DEFINE_string('save_path', '../data/donor_data/',
                    'Path to directory with donor data')
flags.DEFINE_float('trainset_part_percentage', 100.0, 'percentage of dataset')

# Redefine file flags
flags.DEFINE_list(
    'train_data_files', [
        "../data/dataset/train_1.tfrecords",
        "../data/dataset/train_2.tfrecords",
        "../data/dataset/train_3.tfrecords",
        "../data/dataset/train_4.tfrecords"
    ],
    'Training data files in TFRecord format. Multiple files can be passed in a'
    ' comma-separated list. The first file in the list will be used for'
    ' computing the training error.')
flags.DEFINE_string(
    'valid_data_file', '../data/dataset/validation.tfrecords', 'Validation data in TFRecord format.')
flags.DEFINE_string(
    'test_data_file', '../data/dataset/test.tfrecords', 'Testing data in TFRecord format.')
flags.DEFINE_string(
    'sample_data_file', '../data/dataset/sample.tfrecords', 'Sampled batch data in TFRecord format.')

# Model hyperparameters. The default values are exactly what is used during the
# exhaustive evaluation of all models.
flags.DEFINE_integer(
    'batch_size', 256, 'Training batch size.')

FLAGS = flags.FLAGS


def main(*args, **kwargs):
    nasbench = nasbench_api.NASBench(FLAGS.path_to_nasbench)
    module = nasbench.fixed_statistics[FLAGS.hash_key]
    spec = model_spec.ModelSpec(module['module_adjacency'], module['module_operations'])
    
    config = nasbench_config.build_config()
    for flag in FLAGS.flags_by_module_dict()[args[0][0]]:
        config[flag.name] = flag.value
    config['use_tpu'] = False
    config['use_KD'] = False

    
    logging.info("Prepare KD dataset")
    dataset_files = FLAGS.train_data_files + [FLAGS.valid_data_file, FLAGS.test_data_file, FLAGS.sample_data_file]
    prepare_kd_dataset(spec, config, FLAGS.save_path, dataset_files, FLAGS.trainset_part_percentage)
    
    


if __name__ == "__main__":
    tf.enable_eager_execution()
    
    logging.set_verbosity('info')
    logging.set_stderrthreshold('info')

    app.run(main)
