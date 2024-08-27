import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import imageio
import math

matplotlib.use('TkAgg')


def plot_running_avg(values, model_name, plot_name, x_name, y_name, distinct=True, folder_name=None, window=500):
    keys = list(values.keys())
    if folder_name is None:
        folder_name = plot_name
    if distinct:
        for key in keys:
            plot_running_avg({key: values[key]}, model_name, plot_name + '_' + key, x_name, y_name, False, folder_name, window=window)
    plt.figure()
    for key in keys:
        running_average = np.convolve(values[key], np.ones(window), mode='valid') / window
        plt.plot(running_average, label=f'{key} (Running Average)')

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()

    directory = os.path.dirname('plots/' + model_name + '/' + folder_name + '/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('plots/' + model_name + '/' + folder_name + '/' + plot_name + '.png')
    plt.close('all')


def calculate_distances_to_agent(positions):
    agent_positions = positions['agent_0']
    dist1 = [math.dist(x, y) for x, y in zip(agent_positions, positions['adversary_0'])]
    dist2 = [math.dist(x, y) for x, y in zip(agent_positions, positions['adversary_1'])]
    dist3 = [math.dist(x, y) for x, y in zip(agent_positions, positions['adversary_2'])]
    dist_avg = [np.mean(x) for x in zip(dist1, dist2, dist3)]
    return {'adversary_avg': dist_avg, 'adversary_0': dist1, 'adversary_1': dist2, 'adversary_2': dist3}


def plot_avg_distance_to_agent(positions, model_name, plot_name, x_name, y_name, folder_name):
    plot_running_avg(calculate_distances_to_agent(positions), model_name, plot_name, x_name, y_name, window=3, folder_name=folder_name)


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
    plt.close('all')


def create_gif(frame_list, model_name, plot_name):
    frames_np = [image.convert('RGB') for image in frame_list]
    directory = os.path.dirname('plots/' + model_name + '/gifs/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    imageio.mimsave('plots/' + model_name + '/gifs/' + plot_name, frames_np, duration=0.2)


def get_env_name(env, accelerate_agents, distance, time):
    name = env
    if accelerate_agents:
        name += '+accelerate'
    if distance:
        name += '+distance'
    if time:
        name += '+time'
    return name


def get_policy_name(policy, mode, ranom_no_action):
    name = policy
    name += '-' + mode
    if ranom_no_action:
        name += '+rndNoAction'
    return name


def write_time(main_dir, elapsed_time):
    with open('plots/' + main_dir + "/time.txt", "w+") as time_file:
        time_file.write(f"Elapsed time: {elapsed_time:.2f} seconds")


def is_point_in_triangle(p, x, y, z):
    v0 = np.array(z) - np.array(x)
    v1 = np.array(y) - np.array(x)
    v2 = np.array(p) - np.array(x)

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def point_to_line_distance(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    return np.abs(np.cross(b-a, a-p)) / np.linalg.norm(a-b)


def point_distance_to_triangle(p, a, b, c):
    inside = is_point_in_triangle(p, a, b, c)
    d1 = point_to_line_distance(p, a, b)
    d2 = point_to_line_distance(p, b, c)
    d3 = point_to_line_distance(p, c, a)

    min_distance = min(d1, d2, d3)

    return min_distance if inside else -min_distance


def get_triangle_positions(positions):
    distances = []
    for i in range(len(positions['agent_0'])):
        agent_pos = positions['agent_0'][i]
        v0 = positions['adversary_0'][i]
        v1 = positions['adversary_1'][i]
        v2 = positions['adversary_2'][i]

        distances.append(point_distance_to_triangle(agent_pos, v0, v1, v2))
    return distances


def get_triangle_coverage(positions):
    return np.sum([1 if x > 0 else 0 for x in get_triangle_positions(positions)]) / len(positions['agent_0'])


def plot_triangle_position(positions, model_name, plot_name):
    distances = np.array(get_triangle_positions(positions))
    positive_mask = distances > 0
    negative_mask = distances <= 0

    plt.scatter(np.where(positive_mask)[0], distances[positive_mask], color='green', label='Inside (Positive)', s=5)
    plt.scatter(np.where(negative_mask)[0], distances[negative_mask], color='red', label='Outside (Negative)', s=5)
    plt.axhline(0, color='black', linestyle='--', label='Triangle Boundary')
    plt.xlabel('Step')
    plt.ylabel('Distance')
    plt.legend()
    directory = os.path.dirname('plots/' + model_name + '/position_to_triangle/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('plots/' + model_name + '/position_to_triangle/' + plot_name)
    plt.close('all')


def calculate_cooperation(positions):
    adv_pos = [np.array(positions['adversary_0']), np.array(positions['adversary_1']), np.array(positions['adversary_2'])]
    adv_shifts = [x[1:] - x[:-1] for x in adv_pos]
    norms1 = [np.linalg.norm(x - y) for x, y in zip(adv_shifts[0], adv_shifts[1])]
    norms2 = [np.linalg.norm(x - y) for x, y in zip(adv_shifts[1], adv_shifts[2])]
    norms3 = [np.linalg.norm(x - y) for x, y in zip(adv_shifts[2], adv_shifts[0])]
    max_norms = [max(x) for x in zip(norms1, norms2, norms3)]
    return max_norms


def get_cooperation_avg(positions):
    return np.mean(calculate_cooperation(positions))
def plot_cooperation(positions, model_name, plot_name, x_name, y_name, folder_name):
    max_norms = calculate_cooperation(positions)
    plot_running_avg({'adversaries cooperation': max_norms}, model_name, plot_name, x_name, y_name, distinct=False, folder_name=folder_name, window=1)


def calculate_pursuit(positions):
    agent_pos = np.array(positions['agent_0'])
    adv_pos = [np.array(positions['adversary_0']), np.array(positions['adversary_1']), np.array(positions['adversary_2'])]
    agent_shifts = agent_pos[1:] - agent_pos[:-1]
    adv_shifts = [x[1:] - x[:-1] for x in adv_pos]
    norms1 = [np.linalg.norm(x - y) for x, y in zip(adv_shifts[0], agent_shifts)]
    norms2 = [np.linalg.norm(x - y) for x, y in zip(adv_shifts[1], agent_shifts)]
    norms3 = [np.linalg.norm(x - y) for x, y in zip(adv_shifts[2], agent_shifts)]
    max_norms = [max(x) for x in zip(norms1, norms2, norms3)]
    return max_norms


def get_pursuit_avg(positions):
    return np.mean(calculate_pursuit(positions))
def plot_pursuit(positions, model_name, plot_name, x_name, y_name, folder_name):
    max_norms = calculate_pursuit(positions)
    plot_running_avg({'adversaries pursuit accordance': max_norms}, model_name, plot_name, x_name, y_name, distinct=False, folder_name=folder_name, window=1)




