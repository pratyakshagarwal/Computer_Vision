import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics  import confusion_matrix, roc_curve
import seaborn as sns
import tensorflow as tf


# plot a confusion matrix
def plot_comfusion_matrix(labels, predicted, threshold, save=None, filename=None):
    cm = confusion_matrix(labels, predicted > threshold)
    print(cm)

    plt.figure(figsize=(8, 8))

    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix - {}'.format(threshold))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    if save is not None:
        plt.savefig(filename)
        
    plt.show()


# plot a roc curve and if wanted save it
def plot_roc_curve(labels, predicted, save=None, filename=None):
    fp, tp, threshold = roc_curve(labels, predicted)
    print(len(fp))
    print(len(tp))
    print(len(threshold))

    plt.plot(fp, tp)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.grid()

    skip = 25

    for i in range(0, len(threshold), skip):
        plt.text(fp[i], tp[i], threshold[i])

    if save is not None:
        plt.savefig(filename)
    plt.show()