import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Data loading
def split_train_val_nd_test(dataset, train_size, val_size, test_size):
    dataset_size = len(dataset)

    train_dataset = dataset.take(int(dataset_size*train_size))

    val_test_dataset = dataset.skip(int(dataset_size * train_size))
    val_dataset = val_test_dataset.take(int(dataset_size * val_size))

    test_dataset = val_test_dataset.skip(int(dataset_size*val_size))
    return train_dataset, val_dataset, test_dataset


# Data visulization 
def plot_data(dataset, dataset_info):
    for i, (image, label) in enumerate(dataset.take(4)):
        ax = plt.subplot(2, 2, i+1)
        plt.imshow(image)
        plt.title(dataset_info.features['label'].int2str(label))
    plt.show()


# visualize the original and augumented picture 
def visualize(original, augumented, save=None, filename=None):
    plt.subplot(1, 2, 1)
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.imshow(augumented)

    if save is not None:
        plt.savefig(filename)

    plt.show()

# Data processing

img_size = 256
# Resize the image in the desired size 
def resize_nd_rescale(image, label):
    return tf.image.resize(image, (img_size, img_size))/255.0, label

def batching_nd_shuffling(train_dataset, buffer_size=None, batch_size=None, func=None, shuffling=True, mapping=True, batching=True, prefetching=True):
    if shuffling:
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    if mapping:
        train_dataset = train_dataset.map(func)

    if batching:
        train_dataset = train_dataset.batch(batch_size)

    if prefetching:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset

def batching_nd_shuffling_with_parallel_calls(train_dataset, buffer_size=None, batch_size=None, func=None, shuffling=True, mapping=True, batching=True, prefetching=True):
    if shuffling:
        train_dataset = train_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    if mapping:
        train_dataset = train_dataset.map(func, num_parallel_calls=tf.data.AUTOTUNE)

    if batching:
        train_dataset = train_dataset.batch(batch_size)

    if prefetching:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset

if __name__ == '__main__':
    pass