from keras.layers import RandomBrightness, RandomContrast, RandomFlip, RandomRotation
import keras
import tensorflow as tf
import tensorflow_probability as tfp
from Utils.data_preprocessing_utils import resize_nd_rescale, batching_nd_shuffling_with_parallel_calls, batching_nd_shuffling
from Utils.data_augmentation_utils import  augumented_layers

img_size = 256

def resize_nd_rescale(image, label):
    return tf.image.resize(image, (img_size, img_size))/255.0, label


def generate_random_box(lamda):
    """Generate a random bounding box based on CutMix blending parameter."""
    r_x = tf.cast(tfp.distributions.Uniform(0, img_size).sample(1)[0], dtype=tf.int32)
    r_y = tf.cast(tfp.distributions.Uniform(0, img_size).sample(1)[0], dtype=tf.int32)

    r_w = tf.cast(img_size * tf.math.sqrt(1 - lamda), dtype=tf.int32)
    r_h = tf.cast(img_size * tf.math.sqrt(1 - lamda), dtype=tf.int32)

    r_x = tf.clip_by_value(r_x - r_w // 2, 0, img_size)
    r_y = tf.clip_by_value(r_y - r_h // 2, 0, img_size)

    x_b_r = tf.clip_by_value(r_x + r_w // 2, 0, img_size)
    y_b_r = tf.clip_by_value(r_y + r_h // 2, 0, img_size)

    r_w = x_b_r - r_x
    r_h = y_b_r - r_y

    if r_w == 0:
        r_w = 1

    if r_h == 0:
        r_h = 1

    return r_y, r_x, r_h, r_w


def cutmix(train_dataset_1, train_dataset_2):
    """Apply CutMix augmentation to two input images and labels."""
    (image_1, label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

    lamda = tfp.distributions.Beta(0.4, 0.4).sample(1)[0]
    r_y, r_x, r_h, r_w = generate_random_box(lamda)

    crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
    pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, img_size, img_size)

    crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
    pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, img_size, img_size)

    blended_image = image_1 - pad_1 + pad_2

    lamda = tf.cast(1 - (r_w * r_h) / (img_size * img_size), dtype=tf.float32)
    blended_label = lamda * tf.cast(label_1, dtype=tf.float32) + (1 - lamda) * tf.cast(label_2, dtype=tf.float32)

    return blended_image, blended_label


Configurations = {
        'batch_size': 32,
        'img_size': 256,
        'class_names': ['angry', 'happy', 'sad'],
    }


augment_layers = keras.Sequential([
  RandomRotation(factor = (-0.025, 0.025)),
  RandomFlip(mode='horizontal',),
  RandomContrast(factor=0.2),                   
])


def augment_layer(image, label):
  image, label = resize_nd_rescale(image=image, label=label)
  return augment_layers(image, training = True), label

train_dataset_dir = r'C:\Users\praty\OneDrive\Computer_Vision\Human_Emotion_Detection\train'
val_dataset_dir = r'C:\Users\praty\OneDrive\Computer_Vision\Human_Emotion_Detection\test'

train_dataset = keras.utils.image_dataset_from_directory(
        train_dataset_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=Configurations['class_names'],
        color_mode='rgb',
        batch_size=Configurations['batch_size'],
        image_size=(Configurations['img_size'], Configurations['img_size']),
        shuffle=True,
        seed=99,
    )

val_dataset = keras.utils.image_dataset_from_directory(
        val_dataset_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=Configurations['class_names'],
        color_mode='rgb',
        batch_size=1,#CONFIGURATION["BATCH_SIZE"],
        image_size=(Configurations['img_size'], Configurations['img_size']),
        shuffle=True,
        seed=99,
    )

train_dataset = train_dataset.map(resize_nd_rescale)
train_dataset1 = batching_nd_shuffling_with_parallel_calls(train_dataset, batching=False, shuffling=False, prefetching=False,func=augment_layer)
train_dataset2 = batching_nd_shuffling_with_parallel_calls(train_dataset, batching=False, shuffling=False, prefetching=False, func=augment_layer)

mixed_dataset = tf.data.Dataset.zip((train_dataset1, train_dataset2))

train_dataset = batching_nd_shuffling_with_parallel_calls(mixed_dataset, batching=False, shuffling=False, prefetching=True, func=cutmix)
val_dataset = batching_nd_shuffling(val_dataset, batching=False, shuffling=False, func=resize_nd_rescale)