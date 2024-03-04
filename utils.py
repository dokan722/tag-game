import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import imageio
matplotlib.use('TkAgg')


def plot_rewards(training_rewards, model_name, plot_name):
    keys = list(training_rewards[0].keys())
    plt.figure()
    for key in keys:
        values = [d[key] for d in training_rewards]
        running_average = np.cumsum(values) / (np.arange(len(values)) + 1)
        plt.plot(running_average, label=f'{key} (Running Average)')

    plt.xlabel('Data Index')
    plt.ylabel('Values')
    plt.legend()

    directory = os.path.dirname('plots/' + model_name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('plots/' + model_name + '/' + plot_name)


def plot_positions(positions, model_name, plot_name):
    grid_size = 20
    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    positions = np.array(positions)
    heatmap, x_edges, y_edges = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_grid, y_grid])

    plt.figure()
    plt.imshow(heatmap.T, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrRd')
    plt.colorbar(label='Frequency')
    plt.title('Heatmap of Position Frequencies')

    directory = os.path.dirname('plots/' + model_name + '/position_heatmaps/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('plots/' + model_name + '/position_heatmaps/' + plot_name)


def create_gif(frame_list, model_name, plot_name):
    frames_np = [image.convert('RGB') for image in frame_list]
    directory = os.path.dirname('plots/' + model_name + '/gifs/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    imageio.mimsave('plots/' + model_name + '/gifs/' + plot_name, frames_np, duration=0.2)

