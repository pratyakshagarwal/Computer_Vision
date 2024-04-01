import math
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Dropout, InputLayer
from keras.regularizers import L2
from keras.layers import Rescaling, Resizing
from keras import Sequential
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.losses import CategoricalCrossentropy
from tensorboard.plugins.hparams import api as hp
from keras.metrics import RootMeanSquaredError
from keras.callbacks import Callback, CSVLogger
from Utils.metrics import CustomAccuracy, CustomBce
from Utils.callbacks_utils import LossCallback


Configurations = {
        'batch_size': 32,
        'img_size': 256,
        'class_names': ['angry', 'happy', 'sad'],
        'learning_rate': 0.001,
        'n_epochs':10,
        'dropout_rate':0,
        'regularization_rate':0,
        'n_filter1':6,
        'n_filter2':16,
        'kernel_size':3,
        'n_stride':1,
        'pool_size':2,
        'ndense_1':100,
        'ndense_2':10,
        'num_classes':3
    }

resize_rescale_layers = tf.keras.Sequential([
       Resizing(Configurations['img_size'], Configurations['img_size']),
       Rescaling(1./255),                 
])

class CNN_Model(tf.keras.models.Model):
    def __init__(self, dropout_rate=0.2, optimizer=Adam(learning_rate=Configurations['learning_rate']), loss_fn=CategoricalCrossentropy(), metric=CustomAccuracy(), customly=False):

        super(CNN_Model, self).__init__()

        self.customly = customly

        self.model = Sequential()
        # self.model.add(resize_rescale_layers)
        self.model.add(Conv2D(filters=Configurations['n_filter1'],kernel_size=Configurations['kernel_size'],strides=Configurations['n_stride'],
                               activation='relu', padding='valid', kernel_regularizer=L2(Configurations['regularization_rate']),input_shape=(Configurations['img_size'], Configurations['img_size'], 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=Configurations['pool_size'], strides=Configurations['n_stride']*2))
        self.model.add(Dropout(rate=Configurations['dropout_rate']))

        self.model.add(Conv2D(filters=Configurations['n_filter2'], kernel_size=Configurations['kernel_size'], strides=Configurations['n_stride'],
                                activation='relu', padding='valid', kernel_regularizer=L2(Configurations['regularization_rate'])))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=Configurations['pool_size'], strides=Configurations['n_stride']*2))
    
        self.model.add(Flatten())

        self.model.add(Dense(Configurations['ndense_1'], activation='relu', kernel_initializer=he_uniform(), kernel_regularizer=L2(Configurations['regularization_rate']))) 
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=dropout_rate))

        self.model.add(Dense(Configurations['ndense_2'], activation='relu', kernel_initializer=he_uniform(), kernel_regularizer=L2(Configurations['regularization_rate'])))
        self.model.add(BatchNormalization())

        self.model.add(Dense(Configurations['num_classes'], activation='softmax'))

        self.metric = metric
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if customly == True:
            self.metric_val = metric

            current_time = datetime.datetime.now().strftime('%d%m%y - %h%m%s')
            custom_train_dir = './logs/' + current_time + '/custom/train'
            custom_val_dir =   './logs/' + current_time + '/custom/val'

            self.custom_train_writer = tf.summary.create_file_writer(custom_train_dir)
            self.custom_val_writer = tf.summary.create_file_writer(custom_val_dir)
        else:
            self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metric)

        self.HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([16,32,64,128]))
        self.HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([16,32,64,128]))
        self.HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([0.1,0.2,0.3]))
        self.HP_REGULARIZATION_RATE = hp.HParam('regularization_rate', hp.Discrete([0.001,0.01,0.1]))
        self. HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-4, 1e-3]))

    def train_nd_evaluate(self, train_data, val_data=None, epochs=Configurations['n_epochs'], batch_size=Configurations['batch_size'], verbose=1, callbacks=LossCallback()):
        if self.customly:
            for epoch in range(epochs):
                print('Training starts for epoch number {}'.format(epoch))
                for step, (x_batch, y_batch) in enumerate(train_data):
                
                    loss = self.training_block(x_batch, y_batch)

                if verbose == 1:
                    print('Training loss', loss)
                    print('The accuracy is:', self.metric.result())

                with self.custom_train_writer.as_default():
                    tf.summary.scalar('Training Loss', data = loss, step = epoch)
                with self.custom_train_writer.as_default():
                    tf.summary.scalar('Training Accuracy', data = self.metric.result(), step = epoch)

                self.metric.reset_states()

                if val_data != None:
                    for step, (x_batch_val, y_batch_val) in enumerate(val_data):
                 
                        loss_val = self.val_block(x_batch_val, y_batch_val)

                    if verbose == 1:
                        print('Validation loss', loss_val)
                        print('The val accuracy is:', self.metric_val.result())

                    with self.custom_val_writer.as_default():
                        tf.summary.scalar('Training Loss', data = loss_val, step = epoch)
                    with self.custom_val_writer.as_default():
                        tf.summary.scalar('Training Accuracy', data = self.metric_val.result(), step = epoch)
                    self.metric_val.reset_states()

        else:
            if val_data != None:
                history = self.model.fit(train_data, validation_data=val_data, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
            else :
                history = self.model.fit(train_data, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

        if self.customly == False:
            return history

    @tf.function
    def training_block(self, x_batch, y_batch):
        with tf.GradientTape() as recoder:
              y_pred = self.model(x_batch, training=True)
              loss = self.loss_fn(y_batch, y_pred) 
         
        partial_derivatives = recoder.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(partial_derivatives, self.model.trainable_weights))
          
        self.metric.update_state(y_batch, y_pred)
        return loss
    
    @tf.function
    def val_block(self, x_batch_val, y_batch_val):
        y_pred_val = self.model(x_batch_val, training=True)
        loss_val = self.loss_fn(y_batch_val, y_pred_val) 
        self.metric_val.update_state(y_batch_val, y_pred_val)

        return loss_val
    
    def hypertune(self, dataset, img_size):
        
        run_number = 0
        for num_units_1 in self.HP_NUM_UNITS_1.domain.values:
            for num_units_2 in self.HP_NUM_UNITS_2.domain.values:
                for dropout_rate in self.HP_DROPOUT.domain.values:
                    for regularization_rate in self.HP_REGULARIZATION_RATE.domain.values:
                        for learning_rate in self.HP_LEARNING_RATE.domain.values:

                            hparams = {
                                self.HP_NUM_UNITS_1: num_units_1,
                                self.HP_NUM_UNITS_2: num_units_2,
                                self.HP_DROPOUT: dropout_rate,
                                self.HP_REGULARIZATION_RATE: regularization_rate,
                                self.HP_LEARNING_RATE: learning_rate,
              
                            }
                            file_writer = tf.summary.create_file_writer('logs/hparams-' + str(run_number))

                            with file_writer.as_default():
                                hp.hparams(hparams)
                                accuracy = self.model_tune(hparams, dataset=dataset, img_size=img_size)
                                tf.summary.scalar('accuracy', accuracy, step = 0)
                            print("For the run {}, hparams num_units_1:{}, num_units_2:{}, dropout:{}, regularization_rate:{}, learning_rate:{} accuracy is {}".format(run_number, hparams[self.HP_NUM_UNITS_1], hparams[self.HP_NUM_UNITS_2],
                                                             hparams[self.HP_DROPOUT], hparams[self.HP_REGULARIZATION_RATE],
                                                             hparams[self.HP_LEARNING_RATE], accuracy))
                            run_number += 1

    def model_tune(self, hparams, dataset, img_size):
            self.model = tf.keras.Sequential([
                InputLayer(input_shape = (img_size, img_size, 3)),

                Conv2D(filters = 6, kernel_size = 3, strides=1, padding='valid',
                    activation = 'relu',kernel_regularizer = L2(hparams[self.HP_REGULARIZATION_RATE])),
                BatchNormalization(),
                MaxPool2D(pool_size = 2, strides= 2),
                Dropout(rate = hparams[self.HP_DROPOUT]),

                Conv2D(filters = 16, kernel_size = 3, strides=1, padding='valid',
                    activation = 'relu', kernel_regularizer = L2(hparams[self.HP_REGULARIZATION_RATE])),
                BatchNormalization(),
                MaxPool2D(pool_size = 2, strides= 2),

                Flatten(),
    
                Dense( hparams[self.HP_NUM_UNITS_1], activation = "relu", kernel_regularizer = L2(hparams[self.HP_REGULARIZATION_RATE])),
                BatchNormalization(),
                Dropout(rate = hparams[self.HP_DROPOUT]),
    
                Dense(hparams[self.HP_NUM_UNITS_2], activation = "relu", kernel_regularizer = L2(hparams[self.HP_REGULARIZATION_RATE])),
                BatchNormalization(),

                Dense(3, activation = "sigmoid"),
            ])
            self.model.compile(
                optimizer= Adam(learning_rate = hparams[self.HP_LEARNING_RATE]),
                loss=self.loss_fn,
                metrics='accuracy',)

            self.model.fit(dataset, epochs=1)
            _, accuracy = self.model.evaluate(dataset)
            return accuracy

    def evaluate(self, test_data=None):
        print(self.model.evaluate(test_data))

    def get_summary(self):
        return self.model.summary()
    
    def save_model(self, model_name):
        self.model.save(model_name)

if __name__ == '__main__':
    model = CNN_Model()
    # print(model.get_summary())