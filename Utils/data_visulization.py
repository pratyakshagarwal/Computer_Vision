import matplotlib.pyplot as plt
import tensorflow as tf

def plot_model_loss(history, save=None, filename=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.axis('off')

    if save != None:
        plt.savefig(filename)
    plt.show()

def plot_model_accuracy(history, save=None, filename=None):
    plt.plot(history.history['train_accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train_accuracy', 'val_accuracy'])

    if save != None:
        plt.savefig(filename)
    plt.show()

def visualize_data(dataset, class_name, save=None, filename=None):
    plt.figure(figsize=(12, 12))

    for images, labels in dataset.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i+1)
            plt.imshow(images[i]/255.0)
            plt.title(class_name[tf.argmax(labels[i], axis=0).numpy()])
            plt.axis('off')

            if save!=None:
                plt.savefig(filename)
    plt.show()

def visualize_data_after_training(dataset, model, class_names, save=None, filepath=None):
    plt.figure(figsize = (12, 12))
        
    for images, labels in dataset.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i+1)
            plt.imshow(images[i])
            plt.title("True Label - :" + class_names[tf.argmax(labels[i], axis=0).numpy()]+ "\n" + "predicted Label - :" + class_names[tf.argmax(model(tf.expand_dims(images[i], axis=0)), axis=-1).numpy()[0]])
            plt.axis('off')
        if save!=None:
            plt.save(filepath)
        plt.show()