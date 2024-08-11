import numpy as np
import matplotlib.pyplot as plt

def load_csv_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    curves = []
    for id in np.unique(data[:, 0]):
        subset = data[data[:, 0] == id][:, 1:]
        segments = []
        for segment_id in np.unique(subset[:, 0]):
            segment = subset[subset[:, 0] == segment_id][:, 1:]
            segments.append(segment)
        curves.append(segments)
    return curves

def visualize_curves(curves, title='Visualization'):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for index, segments in enumerate(curves):
        color = colors[index % len(colors)]
        for segment in segments:
            ax.plot(segment[:, 0], segment[:, 1], c=color, linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()
