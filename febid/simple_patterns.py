"""
Stream-file reader and pattern generator
"""
import math
import numpy as np
from pandas import read_csv

MAX_UNIT = 65535  # 16-bit stream-fle resolution
STREAM_FILE_DWELL_TIME_UNIT = 1E-7  # s
X_DIM_MIN = 100  # nm
Y_DIM_MIN = 100  # nm
X_DIM_DEFAULT = 200  # nm
Y_DIM_DEFAULT = 200  # nm
Z_DIM_DEFAULT = 200  # nm


def pixel_pitch(hfw):
    """
    Get pixel pitch for a target HFW

    :param hfw: half field width
    :return:
    """
    return hfw * 1000 / MAX_UNIT


def collapse_positions(data_series):
    """
    Sum dwell time of consecutive instructions with identical coordinates

    :param data_series: stream-file
    :return: collapsed stream-file
    """
    if data_series.size == 0:
        return np.array([])

    x, y, d = data_series.T
    mask = np.concatenate(([False], (x[:-1] == x[1:]) & (y[:-1] == y[1:])))

    indices = np.where(~mask)[0]
    collapsed_data = np.column_stack((x[indices], y[indices], np.add.reduceat(d, indices)))

    return collapsed_data


def open_stream_file(file, hfw, offset=200, collapse=False):
    """
    Open stream file, convert to nm and define enclosing volume dimensions.
        A valid stream-file should consist of 3 columns and start with 's16' line.

    :param file: path to the stream-file
    :param hfw: target half field width
    :param offset: determines a margin around the printing path
    :param collapse: if True, summ dwell time of consecutive instructions with identical coordinates
    :return: normalized directives in nm and s, dimensions of the enclosing volume [z, y, x] in nm
    """
    # Opening file and parsing text
    # Default columns:
    # 0 – Dwell time
    # 1 – x-position
    # 2 – y-position
    try:
        with open(file, mode='r+', encoding='utf-8', errors='ignore') as f:
                text = f.readline()
                if text != 's16\n':
                    raise IOError('Not a valid stream file!')
                delim = ' '  # delimiter between the columns
                header = 3  # number of lines to skip in the beginning
                columns = (0, 1, 2)  # numbers of columns to get
                print(f'Reading stream-file {file}  ...')
                data = read_csv(file, dtype=np.float64, comment='#', delimiter=delim, skiprows=header, usecols=columns).to_numpy()
                print('Done!')
    except UnicodeDecodeError:
        raise FileNotFoundError('Corrupted stream file!')
    except FileNotFoundError:
        raise FileNotFoundError('File not found!')

    # Converting to simulation units
    unit_pitch = pixel_pitch(hfw)  # nm/pixel
    data[:, 0] *= STREAM_FILE_DWELL_TIME_UNIT  # converting [0.1 µs] to [s]
    data[:, 1] *= unit_pitch  # converting stream-file units to [nm]
    data[:, 2] *= unit_pitch
    # Determining volume dimensions with an offset
    x_max, x_min = data[:, 1].max(), data[:, 1].min()
    x_dim = (x_max - x_min) + offset
    x_delta = offset / 2
    y_max, y_min = data[:, 2].max(), data[:, 2].min()
    y_dim = (y_max - y_min) + offset
    y_delta = offset / 2
    z_dim = Z_DIM_DEFAULT

    if x_dim < X_DIM_MIN or y_dim < Y_DIM_MIN:  # checking if both dimensions are at least 100 nm
        if x_dim < X_DIM_MIN:
            x_dim = X_DIM_DEFAULT
            x_delta = x_dim / 2
        if y_dim < Y_DIM_MIN:
            y_dim = Y_DIM_DEFAULT
            y_delta = y_dim / 2
    # Getting local coordinates
    data[:, 1] -= x_min - x_delta  # shifting path center to the center of the volume
    data[:, 2] -= y_min - y_delta  # shifting path center to the center of the volume
    data = np.roll(data, -1, 1)

    # Summing dwell time of consecutive points with same coordinates
    if collapse:
        data = collapse_positions(data)

    return data, np.array([z_dim, y_dim, x_dim], dtype=int)


def analyze_pattern(file, hfw):
    """
    Show stream-file total time and distinct patterning speeds

    :param file: path to the stream-file
    :param hfw: target half field width
    """
    """
    Parse stream-file and split it into stages
    """
    unit_pitch = pixel_pitch(hfw)
    data, shape = open_stream_file(file, hfw=hfw, collapse=True)
    stages = []
    total_time = data[:, 2].sum()
    delta = data[1:] - data[:-1]
    i = 0
    while i < delta.shape[0]:
        if data[i, 2] > 1:  # considering 1 nm/s the lowest patterning speed and everything above is stationary
            stages.append(('Stationary', data[i, 2]))
            i += 1
        else:
            t = 0
            pos0 = np.array([data[i, 0], data[i, 1]])  # x, y distance
            pos = np.array([0, 0])
            flag_end = False
            while i < delta.shape[0] - 1:
                t += data[i, 2]
                if np.isclose(delta[i, 0:2], delta[i + 1, 0:2], atol=unit_pitch).all() and delta[i, 2] == delta[
                    i + 1, 2]:
                    i += 1
                else:
                    t += data[i + 1, 2]
                    flag_end = True  # telling the loop, that termination was due to the end of a stage
                    break
            else:
                if not flag_end:  # if the loop exited due to array bounds, collect remaining cells
                    t += data[i, 2]
                    i += 1
                    t += data[i, 2]
                    flag_end = False
                pos[:] = data[i, 0], data[i, 1]
                lp = pos - pos0
                l = math.sqrt(lp[0] ** 2 + lp[1] ** 2)
                speed = round(l / t)
                stages.append((f'{speed} nm/s', t))
                i += 1
    print(f'Total patterning time: {total_time:.4f} s')
    print('Stages: |', end='')
    for stage, d in stages:
        print(f'{stage}, {d:.4f} | ', end='')
    else:
        print(' ')


def generate_pattern(pattern, loops, dwell_time, x, y, params=(1,1), step=1):
    """
    Generate a stream-file for a simple figure.

    :param pattern: name of a shape: point, line, square, rectangle, circle
    :param loops: number of passes
    :param dwell_time: time spent on each point, s
    :param x: center x position of the figure, nm
    :param y: center y position of the figure, nm
    :param params: figure parameters, nm;
        (length) for line, (diameter) for circle, (edge length) for cube
    :param step: distance between each point, nm
    :return: array(x positions[nm], y positions[nm], dwell time[s])
    """
    pattern = pattern.casefold()
    pattern_functions = {
        'point': generate_point,
        'line': generate_line,
        'circle': generate_circle,
        'square': generate_square,
        'rectangle': generate_square
    }
    try:
        path = pattern_functions[pattern](loops, dwell_time, x, y, *params, step=step)
    except KeyError:
        raise ValueError(f'Unknown pattern: {pattern}')
    return path


def generate_point(loops, dwell_time, x, y, a, b, step=1):
    """
    Generate a pattern of a point figure.

    :param loops: number of passes
    :param dwell_time: time spent on a point, s
    :param x: x position of the point, nm
    :param y: y position of the point, nm
    :return: array(x positions[nm], y positions[nm], dwell time[s])
    """
    path = np.asarray([x, y, loops * dwell_time]).reshape(1, 3)
    return path


def generate_circle(loops, dwell_time, x, y, diameter, _, step=1):
    """
    Generate a pattern of a circle figure.

    :param loops: number of passes
    :param dwell_time: time spent on each point, s
    :param x: center x position of the figure, nm
    :param y: center y position of the figure, nm
    :param diameter: circle diameter, nm
    :param step: distance between each point, nm
    :return: array(x positions[nm], y positions[nm], dwell time[s])
    """
    angle_step = step / diameter / 2
    n = int(np.pi * 2 // angle_step)
    loop = np.empty((n, 3))
    stub = np.linspace(angle_step, np.pi * 2, n)
    loop[:, 0] = diameter / 2 * np.sin(stub) + x
    loop[:, 1] = diameter / 2 * np.cos(stub) + y
    loop[:, 2] = dwell_time
    path = np.tile(loop, (loops, 1))
    return path


def generate_square(loops, dwell_time, x, y, side_a, side_b=None, step=1):
    """
    Generate a pattern of a rectangle figure.

    :param loops: number of passes
    :param dwell_time: time spent on each point, s
    :param x: center x position of the figure, nm
    :param y: center y position of the figure, nm
    :param side_a: side length, nm
    :param side_b: side length, nm
    :param step: distance between each point, nm
    :return: array(x positions[nm], y positions[nm], dwell time[s])
    """
    if side_b is None:
        side_b = side_a
    top_left = (y + side_b / 2, x - side_a / 2)
    top_right = (y + side_b / 2, x + side_a / 2)
    low_right = (y - side_b / 2, x + side_a / 2)
    low_left = (y - side_b / 2, x - side_a / 2)
    steps_a = int(side_a / step) - 1
    steps_b = int(side_b / step) - 1
    edge_top = np.empty((steps_a, 3))
    edge_top[:, 1] = np.arange(top_left[1] + step, top_right[1], step)
    edge_top[:, 0] = top_left[0]
    edge_right = np.empty((steps_b, 3))
    edge_right[:, 0] = np.arange(top_right[0] - step, low_right[0], -step)
    edge_right[:, 1] = top_right[1]
    edge_bottom = np.empty((steps_a, 3))
    edge_bottom[:, 1] = np.arange(low_right[1] - step, low_left[1], -step)
    edge_bottom[:, 0] = low_right[0]
    edge_left = np.empty((steps_b, 3))
    edge_left[:, 0] = np.arange(low_left[0] + step, top_left[0], step)
    edge_left[:, 1] = low_left[1]
    path = np.vstack([edge_top, edge_right, edge_bottom, edge_left])
    path[:, 2] = dwell_time
    path = np.tile(path, (loops, 1))
    return path


def generate_line(loops, dwell_time, x, y, line, _, step=1):
    """
    Generate a pattern of a line figure.

    :param loops: number of passes
    :param dwell_time: time spent on each point, s
    :param x: center x position of the figure, nm
    :param y: center y position of the figure, nm
    :param line: line length, nm
    :param step: distance between each point, nm
    :return: array(x positions[nm], y positions[nm], dwell time[s])
    """
    start = x - line / 2
    end = x + line / 2
    loop1 = np.arange(start, end + 1e-8, step)
    loop2 = np.arange(loop1[-1], start - 1e-8, -step)
    loop = np.concatenate([loop1, loop2])
    path = np.empty((loop1.shape[0] + loop2.shape[0], 3))
    path[:, 0] = loop
    path[:, 1] = y
    path[:, 2] = dwell_time
    path = np.tile(path, (loops, 1))
    return path
