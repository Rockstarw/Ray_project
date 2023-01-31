import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import argparse
import time
import sys

# Parallel Code
sys.path.append('/gpfs/projects/nct00/nct00002/cifar-utils')
from cifar import load_cifar

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--n_gpus', type=int, default=1)

world_size = 2       # usually number of GPUs you are using

args = parser.parse_args()
batch_size = args.batch_size * world_size        # update this to the global batch-size
epochs = args.epochs
n_gpus = args.n_gpus

train_ds, test_ds = load_cifar(batch_size)

device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split("e:")[1] for d in devices]

strategy = tf.distribute.MultiWorkerMirroredStrategy()           # update this to be MultiWorkerMirroredStrategy() 


with strategy.scope():
     model = tf.keras.applications.resnet_v2.ResNet50V2(
             include_top=True, weights=None,
             input_shape=(128, 128, 3), classes=10)
     opt = tf.keras.optimizers.SGD(0.01*n_gpus)
     model.compile(loss='sparse_categorical_crossentropy', 
                   optimizer=opt, metrics=['accuracy'])

model.fit(train_ds, epochs=epochs, verbose=2)