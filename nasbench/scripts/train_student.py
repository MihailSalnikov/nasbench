from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
from absl import app
from absl import flags
from absl import logging

import json
from pathlib import Path

from nasbench.lib import config as nasbench_config
from nasbench.lib import model_spec
from nasbench.lib import cifar
from nasbench.lib import model_builder
from nasbench import api as nasbench_api
from nasbench.scripts.run_evaluation import NumpyEncoder

import numpy as np
import tensorflow as tf

from utils import train


# Clear some flags:
for name in [
    'train_data_files', 'valid_data_file',
    'test_data_file', 'sample_data_file',
    'batch_size', 'train_epochs']:
        delattr(flags.FLAGS, name)
        
flags.DEFINE_string('path_to_nasbench', '../data/nasbench_only108.tfrecord',
                    'Path to nasbanch dataset tfrecord files with logits for KD')
flags.DEFINE_string('hash_keys', '../data/student_data/keys_20.json',
                    'Path to JSON array with hash keys of nasbench for traning with KD')
flags.DEFINE_string('save_path', '../data/student_data/',
                    'Path to directory for storing data of students')
flags.DEFINE_float('imitation_lmb', 0.7,
                   'Immitation lambda for KD process')
flags.DEFINE_float('temperature', 20.0,
                   'Temperature for KD process')

# Redefine file flags
flags.DEFINE_list(
    'train_data_files', [
        "../data/dataset/train_1_KD_11.tfrecords",
        "../data/dataset/train_2_KD_11.tfrecords",
        "../data/dataset/train_3_KD_11.tfrecords",
        "../data/dataset/train_4_KD_11.tfrecords"
    ],
    'Training data files in TFRecord format. Multiple files can be passed in a'
    ' comma-separated list. The first file in the list will be used for'
    ' computing the training error.')
flags.DEFINE_string(
    'valid_data_file', '../data/dataset/validation_KD.tfrecords', 'Validation data in TFRecord format.')
flags.DEFINE_string(
    'test_data_file', '../data/dataset/test_KD.tfrecords', 'Testing data in TFRecord format.')
flags.DEFINE_string(
    'sample_data_file', '../data/dataset/sample_KD.tfrecords', 'Sampled batch data in TFRecord format.')

# Model hyperparameters. The default values are exactly what is used during the
# exhaustive evaluation of all models.
flags.DEFINE_integer(
    'batch_size', 256, 'Training batch size.')
flags.DEFINE_integer(
    'train_epochs', 108,
    'Maximum training epochs. If --train_seconds is reached first, training'
    ' may not reach --train_epochs.')

FLAGS = flags.FLAGS


def main(*args, **kwargs):
    with open(FLAGS.hash_keys, "r") as f:
        student_hash_keys = json.load(f)
    
    nasbench = nasbench_api.NASBench(FLAGS.path_to_nasbench)
    for student_key in student_hash_keys:
        module = nasbench.fixed_statistics[student_key]
        spec = model_spec.ModelSpec(module['module_adjacency'], module['module_operations'])
    
        config = nasbench_config.build_config()
        for flag in FLAGS.flags_by_module_dict()[args[0][0]]:
            config[flag.name] = flag.value
        config['use_tpu'] = False
        config['use_KD'] = True
        config['intermediate_evaluations'] = ['1.0']
    
        logging.info("Train and evaluate with config\n{}\n and spec\n{}".format(config, spec))
        save_path = str(Path(FLAGS.save_path, f"student_{student_key}"))
        meta = train(spec, config, save_path)
        
        with open(Path(save_path, "kd_meta.json"), "w") as f:
            json.dump({
                "imitation_lmb": FLAGS.imitation_lmb,
                "temperature": FLAGS.temperature,
            }, f)
        logging.info(f"model {student_key} trained and stored to {save_path}")
    
    


if __name__ == "__main__":
    logging.set_verbosity('info')
    logging.set_stderrthreshold('info')

    app.run(main)
