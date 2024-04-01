import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

class LossCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        print('\n For Epoch Number {} the Model has a Loss of {}'.format(epoch+1, logs['loss']))

    # def on_batch_end(self, epoch, logs):
    #     print('\n For Batch Number {} the Model has a Loss of {}'.format(epoch+1, logs))


class CustomCSVLogger:
    def make_callback(filename='logs.csv', separator=',', append=False):
        csv_callback = CSVLogger(filename=filename, separator=separator, append=append)
        return csv_callback


class CustomEarlyStopping:
    def make_callback(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto', baseline=0, restore_best_weights=False):
        es_callback = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
        return es_callback
    

current_time = datetime.datetime.now().strftime('%d%m%y - %H%M%S')
metric_dir = './logs/' + current_time + '/metrics'
train_writer = tf.summary.create_file_writer(metric_dir)
def scheduler(epoch, lr):
    if epoch < 2:
        Learning_Rate = lr
    else :
        Learning_Rate = lr * tf.math.exp(-0.1)
        Learning_Rate.numpy()
    with train_writer.as_default():
        tf.summary.scalar('Learning Rate', data=Learning_Rate, steps=epoch) 

    return Learning_Rate

class CustomLearningRateScheduler:
    def make_lrs(verbose=1):
        lr_scheduler = LearningRateScheduler(scheduler, verbose=verbose)
        return lr_scheduler


class CustomModelCheckPoint:
    def make_callback(filename='checkpoints/', monitor='val_loss', verbose=0, save_best_only=True,
                      save_weights_only=False, mode='auto', save_freq='epoch'):
        checkpoint_callback = ModelCheckpoint(filepath=filename, monitor=monitor, verbose=verbose, save_best_only=save_best_only, 
                                              save_weights_only=save_weights_only, mode=mode, save_freq=save_freq)
        
        return checkpoint_callback
    

class CustomReduceLROnPlateau:
    def make_callback(monitor='val_accuracy', factor=0.1, patience=2, verbose=1):
        rop_callback = ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=verbose)
        return rop_callback
    
class CustomTensorflowLogs:
    def __init__(self, log_dir=None):
        current_time = datetime.datetime.now().strftime('%d%m%y - %H%M%S')
        if log_dir==None:
            self.lod_dir = './logs/' + current_time
        else :
            self.lod_dir = log_dir + current_time
        
    def make_callback(self):
        tensorflow_callback = TensorBoard(log_dir=self.lod_dir)
        return tensorflow_callback
    

class LogImagesCallbackTensorBoard(Callback):
  def __init__(self, train_data, model):
      super().__init__()
      self.train_data = train_data
      self.model = model

  def on_epoch_end(self, epoch, logs):
    labels = []
    inp = []

    for x,y in self.train_data.as_numpy_iterator():
      labels.append(y)
      inp.append(x)
    labels = np.array([i[0] for i in labels])
    predicted = self.model.predict(np.array(inp)[:,0,...])

    threshold = 0.5

    cm = confusion_matrix(labels, predicted > threshold)
    
    plt.figure(figsize=(8,8))

    sns.heatmap(cm, annot=True,)
    plt.title('Confusion matrix - {}'.format(threshold))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png')

    image = tf.image.decode_png(buffer.getvalue(), channels=3)
    image = tf.expand_dims(image, axis = 0)

    current_time = datetime.datetime.now().strftime('%d%m%y - %h%m%s')
    image_dir = './logs/' + current_time + '/images'
    image_writer = tf.summary.create_file_writer(image_dir)
    
    with image_writer.as_default():
      tf.summary.image("Training data", image, step = epoch)