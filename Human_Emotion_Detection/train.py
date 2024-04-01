import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import CSVLogger
from keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from keras.layers import Rescaling, Resizing
from data import train_dataset, val_dataset, Configurations, augment_layer
from Utils.data_visulization import visualize_data
from Utils.data_visulization import plot_model_accuracy, plot_model_loss
from Utils.callbacks_utils import CustomCSVLogger, CustomEarlyStopping
from model import CNN_Model

if __name__ == '__main__':

    # Visulize the train dataset 
    # visualize_data(train_dataset, Configurations['class_names'])

    metrics = [CategoricalAccuracy(name='accuracy'), TopKCategoricalAccuracy(name='topkaccuracy')]
    
    csv_callback = CSVLogger(filename='Human_Emotion_Detection\logs.csv', separator=',', append=False)

    model = CNN_Model(metric=metrics)
    print(model.get_summary())
    history = model.train_nd_evaluate(train_data=train_dataset, val_data=val_dataset, callbacks=[csv_callback])
    # model.hypertune(train_dataset, img_size=256)
    model.save_model(model_name='Human_Emotion_Detection\emotion_detection.keras')
    plot_model_loss(history)
    plot_model_accuracy(history)