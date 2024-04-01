import pandas as pd
import numpy as np 
import tensorflow as tf
import seaborn as sns
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix
from keras.metrics import Precision, Recall, AUC, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, BinaryAccuracy
from Maleria_Diagonosis.model import CNN_Model
from Utils.metrics import CustomAccuracy, CustomBce
from Utils.data_augmentation_utils import process_data
from Utils.callbacks_utils import LossCallback, CustomCSVLogger, CustomEarlyStopping,  CustomTensorflowLogs
from Utils.data_preprocessing_utils import split_train_val_nd_test, resize_nd_rescale, batching_nd_shuffling


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
    output = 1
    batch_size = 32

    def resize_nd_rescale(image, label):
        return tf.image.resize(image, (img_size, img_size))/255.0, label

    metrics = [BinaryAccuracy(name='accu'), TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'),  FalseNegatives(name='fn'), Precision(name='pr'), Recall(name='rc'), AUC(name='auc')]

    train_dataset, val_dataset, test_dataset = split_train_val_nd_test(dataset[0], train_size, val_size, test_size)
    
    train_dataset = batching_nd_shuffling(train_dataset, buffer_size=8, batch_size=batch_size, func=process_data)
    val_dataset = batching_nd_shuffling(val_dataset, buffer_size=8, batch_size=batch_size, func=resize_nd_rescale)
    
    test_dataset = test_dataset.map(resize_nd_rescale).batch(1)

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)
    csv_callback = CustomCSVLogger.make_callback()
    early_callback  = CustomEarlyStopping.make_callback()
    tensorflow_callbacks = CustomTensorflowLogs(log_dir='Maleria_Diagonosis/logs/').make_callback()

    model = CNN_Model(metric=metrics)

    print(model.get_summary())

    # finding best parameters for the model
    # model.hypertune(val_dataset)

    # training the model
    history = model.train_nd_evaluate(train_dataset, val_dataset, epochs=10, callbacks=[tensorflow_callbacks, csv_callback, early_callback])
    
    # Evaluate the model
    model.evaluate(test_dataset)

    # Save the model
    model.save_model('Maleria_Diagonosis\malerial_diagnosis')
