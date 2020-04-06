from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging

from nasbench.lib import cifar
from nasbench.lib import model_builder
from nasbench.lib import training_time
from nasbench.lib.evaluate import _TrainAndEvaluator
import numpy as np


from nasbench.lib import evaluate
from nasbench.lib import model_spec
from nasbench.lib import config as _config
from nasbench.lib.cifar import _preprocess, _parser

from nasbench.scripts.run_evaluation import NumpyEncoder

import json
import functools
from pathlib import Path

import tensorflow as tf


def _set_batch_dimension(batch_size, images, labels):
    images.set_shape(images.get_shape().merge_with(
      tf.TensorShape([batch_size, None, None, None])))
    labels.set_shape(labels.get_shape().merge_with(
      tf.TensorShape([batch_size])))

    return images, labels

def _dummy_imput_fn(params):
    batch_size = params['batch_size']
    dataset = tf.data.TFRecordDataset(params['file'])
    dataset = dataset.prefetch(buffer_size=batch_size)

   
   # Parse, preprocess, and batch images
    parser_fn = functools.partial(_parser, False)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            parser_fn,
            batch_size=batch_size,
            num_parallel_batches=None,
            drop_remainder=True))

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(_set_batch_dimension, batch_size))

    # Prefetch to overlap in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    

    return dataset

def train(spec, config, save_path):
    evaluator = _TrainAndEvaluator(spec, config, save_path)
    meta = evaluator.run()
    with tf.io.gfile.GFile(str(Path(save_path, "meta.json")), 'w') as f:
        json.dump(meta, f, cls=NumpyEncoder)
    return meta


def prepare_kd_dataset(spec, config, model_path, dataset_files, trainset_part_percentage):
    for filename in dataset_files:  
        raw_dataset = tf.data.TFRecordDataset([filename])
        params = {'file': filename, 'use_KD': False}
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_builder.build_model_fn(spec, config, None),
            config=tf.contrib.tpu.RunConfig(model_dir=model_path),
            params=params,
            train_batch_size=config['batch_size'],
            eval_batch_size=config['batch_size'],
            predict_batch_size=100)

        est_preds = estimator.predict(input_fn=_dummy_imput_fn, yield_single_examples=False)
        all_pred_logits_aug = []
        for preds in est_preds:
            all_pred_logits_aug.append(preds['logits'])
        if len(all_pred_logits_aug) == 0:
            logging.error("all_pred_logits_aug is empty")
        all_pred_logits_aug = np.vstack(all_pred_logits_aug)


        filename = Path(filename)
        name_postfix = '_KD'
        if trainset_part_percentage != 100:
            name_postfix += '_'+str(trainset_part_percentage)
        out_file = filename.with_name(filename.stem+name_postfix)
        out_file = out_file.with_suffix(".tfrecords")
        filename = str(filename)
        with tf.io.TFRecordWriter(str(out_file)) as record_writer:
            for i, raw_record in enumerate(raw_dataset):
                if i >= 10000 * (trainset_part_percentage / 100.0):
                    break
                
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                img = example.features.feature['image']
                label = example.features.feature['label']
                if len(all_pred_logits_aug) <= i:
                    logging.error("all_pred_logits_aug is not anought ({}); i={}".format(len(all_pred_logits_aug), i))
                    continue
                preds = all_pred_logits_aug[i]
                new_label = np.hstack((label.int64_list.value[0], preds))
                feat_preds = tf.train.Feature(float_list=tf.train.FloatList(value=new_label))
                example = tf.train.Example(features=tf.train.Features(
                          feature={
                              'image': img,
                              'label': feat_preds
                          }))
                record_writer.write(example.SerializeToString())
            logging.info("{} stored".format(out_file))
