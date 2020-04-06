from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from nasbench.lib import cifar
from nasbench.lib import model_builder
from nasbench.lib import training_time
import numpy as np
import tensorflow as tf

from nasbench.lib import evaluate
from nasbench.lib import model_spec
from nasbench.lib import config as _config

from nasbench.scripts.run_evaluation import NumpyEncoder

import json

RESULTS_FILE = 'results.json'


import os
os.environ['AUTOGRAPH_VERBOSITY'] = '10'


def main():
    config = _config.build_config()
    config['use_tpu'] = False
    config['train_data_files'] = ['train_1.tfrecords', 'train_2.tfrecords', 'train_3.tfrecords', 'train_4.tfrecords']
    config['valid_data_file'] = ['validation.tfrecords']
    config['test_data_file'] = ['test.tfrecords']
    config['sample_data_file'] = ['sample.tfrecords']
    config['train_epochs'] = 1
    config['use_KD'] = False
    
    matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
          [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
          [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
          [0, 0, 0, 0, 0, 0, 0]]   # output layer
    # Operations at the vertices of the module, matches order of matrix
    labels=['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
    

    matrix = np.array(matrix)
    labels = np.array(labels)

    spec = model_spec.ModelSpec(matrix, labels)
    model_dir = '../data/tmp'

    meta = evaluate.train_and_evaluate(spec, config, model_dir)

    output_file = os.path.join(model_dir, RESULTS_FILE)
    with tf.gfile.Open(output_file, 'w') as f:
      json.dump(meta, f, cls=NumpyEncoder)

    print('OK')
    print(spec.__dict__)

if __name__ == '__main__':
  main()
  # with tf.Session() as sess:
  #   devices = sess.list_devices()
  #   print(devices)