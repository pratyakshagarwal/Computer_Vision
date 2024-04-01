import tensorflow as tf
from keras.preprocessing import image
from keras import Sequential
from keras.layers import Layer
from keras.layers import RandomRotation, RandomFlip, Resizing, Rescaling
import tensorflow_probability as tfp
import albumentations as A
from keras.applications.inception_v3 import preprocess_input
from Utils.data_preprocessing_utils import visualize, resize_nd_rescale

img_size = 224
def augumented(image, label):
    image, label = resize_nd_rescale(image, label)

    image = tf.image.rot90(image, k = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32))
    image = tf.image.stateless_random_saturation(image, 0.3, 0.5)
    image = tf.image.stateless_random_flip_left_right(image)

    return image, label


def mixup(train_dataset_1, train_dataset_2):
    (image_1, label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

    lamda = tfp.distributions.Beta(0.4, 0.4)
    lamda = lamda.sample(1)[0]

    image = lamda * image_1 + (1 - lamda) * image_2
    label = lamda * tf.cast(label_1, dtype=tf.float32) + (1 - lamda) * tf.cast(label_2, dtype=tf.float32)

    return image, label

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


transforms = A.Compose(
    [
      A.Resize(img_size, img_size),

      A.OneOf([A.HorizontalFlip(),
                A.VerticalFlip(),], p = 0.3),
      
      A.RandomRotate90(),   
      #A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.2,
                                contrast_limit=0.2,
                                always_apply=False, p=0.5),
      #A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
      A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
])

# @tf.function
def aug_albument(image):
  data = {"image":image}
  image = transforms(**data)
  image = image["image"]
  image = tf.cast(image/255., tf.float32)
  return image


def process_data(image, label):
    aug_img = tf.numpy_function(func=aug_albument, inp=[image], Tout=tf.float32)
    return aug_img, label


class RotNinety(Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, image):
        return tf.image.rot90(image, k = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32))


class Flip(Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, image):
        return tf.image.stateless_random_flip_left_right(image)
    

augumented_layers = Sequential([
    RotNinety(),
    Flip(), 
])
