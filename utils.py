import time
import matplotlib.pyplot as plt
import seaborn


def start_time():
    return time.time()


def end_time():
    return time.time()


def plot_confusion_matrix(data, labels, output_filename):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title("Confusion Matrix")
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.show()
