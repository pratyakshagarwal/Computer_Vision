import tensorflow as tf
import keras
from keras.models import load_model
from Utils.data_visulization import visualize_data_after_training
from Utils.data_preprocessing_utils import resize_nd_rescale, batching_nd_shuffling
from Utils.evaluation_utils import plot_comfusion_matrix, plot_roc_curve

if __name__ == '__main__':

    val_dataset_dir = r'C:\Users\praty\OneDrive\Computer_Vision\Human_Emotion_Detection\test'
    model = load_model('emotion_detection')
    Configurations = {
        'batch_size': 32,
        'img_size': 256,
        'class_names': ['angry', 'happy', 'sad'],
    }
    threshold = 0.5

    val_dataset = keras.utils.image_dataset_from_directory(
        val_dataset_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=Configurations['class_names'],
        color_mode='rgb',
        batch_size=16,#CONFIGURATION["BATCH_SIZE"],
        image_size=(Configurations['img_size'], Configurations['img_size']),
        shuffle=True,
        seed=99,
    )

    val_dataset = batching_nd_shuffling(val_dataset, batching=False, shuffling=False, func=resize_nd_rescale)

    def prediction_pipeline(img, model, class_names):
        img = tf.constant(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        prediction = class_names[tf.argmax(model(img), axis=-1).numpy()[0]]
        return prediction


    plot_comfusion_matrix()
    visualize_data_after_training(val_dataset, model, Configurations['class_names'])