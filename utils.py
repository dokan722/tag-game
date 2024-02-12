import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def plot_rewards(training_rewards, name):
    keys = list(training_rewards[0].keys())
    plt.figure()
    for key in keys:
        values = [d[key] for d in training_rewards]
        running_average = np.cumsum(values) / (np.arange(len(values)) + 1)
        plt.plot(running_average, label=f'{key} (Running Average)')

    plt.xlabel('Data Index')
    plt.ylabel('Values')
    plt.legend()

    plt.savefig('plots/' + name)


def plot_positions(positions, name):
    grid_size = 50
    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    positions = np.array(positions)
    heatmap, x_edges, y_edges = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_grid, y_grid])

    plt.figure()
    plt.imshow(heatmap.T, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrRd')
    plt.colorbar(label='Frequency')
    plt.title('Heatmap of Position Frequencies')
    plt.savefig('plots/position_heatmaps/' + name)