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

def main():

    config = _config.build_config()
    config['use_tpu'] = False
    config['train_epochs'] = 1
    
    matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
          [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
          [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
          [0, 0, 0, 0, 0, 0, 0]]   # output layer
    # Operations at the vertices of the module, matches order of matrix
    labels=['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
    

    model_dir = '../my_model_dir'
    print(tf.train.latest_checkpoint(model_dir))
    sess = tf.Session()
    saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(model_dir)+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    


    print('OK')

if __name__ == '__main__':
  main()