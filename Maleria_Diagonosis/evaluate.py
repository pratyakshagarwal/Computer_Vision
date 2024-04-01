import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
import tensorflow_datasets as tfds
from sklearn.metrics  import confusion_matrix, roc_curve
import seaborn as sns
from keras.models import load_model
from Utils.data_preprocessing_utils import split_train_val_nd_test, resize_nd_rescale
from Utils.evaluation_utils import plot_roc_curve, plot_comfusion_matrix

if __name__ == '__main__':
    dataset, dataset_info = tfds.load('malaria', with_info=True,
                                  as_supervised=True, 
                                  shuffle_files = True, 
                                  split=['train'])

    train_size = 0.7
    val_size = 0.2
    test_size = 0.1

    img_size = 224
    input_shape = (224, 224)

    def resize_nd_rescale(image, label):
        return tf.image.resize(image, (img_size, img_size))/255.0, label

    train_dataset, val_dataset, test_dataset = split_train_val_nd_test(dataset[0], train_size, val_size, test_size)
    test_dataset = test_dataset.map(resize_nd_rescale)

    test_dataset = test_dataset.batch(1)
    lesnet_model = load_model('Maleria_Diagonosis\malerial_diagnosis')

    labels = []
    input_img = []
    for x, y in test_dataset.as_numpy_iterator():
        labels.append(y)
        input_img.append(x)

    labels = np.array([i[0] for i in labels])
    print(labels)

    predicted = lesnet_model.predict(np.array(input_img)[:, 0, ])
    print(predicted.shape)

    threshold = 0.72
    filename = f'Maleria_Diagonosis\\plots\\confusion_matrix{threshold}.png'

    plot_comfusion_matrix(labels, predicted, threshold, save=1, filename=filename)
    plot_roc_curve(labels, predicted, save=1, filename='Maleria_Diagonosis\\plots\\roc_curve')
    