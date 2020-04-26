from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(x, y, filename):
    with tf.python_io.TFRecordWriter(str(filename)) as record_writer:
        for i in range(len(x)):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(x[i].tobytes()),
                    'label': _int64_feature(y[i])
                }
            ))
            record_writer.write(example.SerializeToString())


if __name__ == "__main__":
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=0.1)

    x_sample, y_sample = x_valid[:100], y_valid[:100]

    base_path = Path("../data/dataset/mnist")
    base_path.mkdir(parents=True, exist_ok=True)

    batch_size = x_train.shape[0] // 4
    for batchid in range(4):
        _x = x_train[batchid*batch_size:(batchid+1)*batch_size]
        _y = y_train[batchid*batch_size:(batchid+1)*batch_size]
        convert_to_tfrecord(
            _x, _y,
            base_path / "train_{}.tfrecords".format(batchid)
        )

    convert_to_tfrecord(x_valid, y_valid, base_path / "valid.tfrecords")
    convert_to_tfrecord(x_test, y_test, base_path / "test.tfrecords")
    convert_to_tfrecord(x_sample, y_sample, base_path / "sample.tfrecords")
