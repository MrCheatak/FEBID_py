"""
Stream file generator
"""
import tkinter.filedialog as fd
import numpy as np


def open_stream_file(file=None, offset=2, collapse=False):
    """
    Open stream file, convert to nm and define enclosing volume dimensions.
        A valid stream-file should consist of 3 columns and start with 's16' line.

    :param file: path to the stream-file
    :param offset: determines a margin around the printing path
    :param collapse: if True, summ dwell time of consecutive instructions with identical coordinates
    :return: normalized directives in nm and s, dimensions of the enclosing volume in nm
    """
    if not file:
        raise FileNotFoundError
        # print('Specify a stream file:')
        # file = fd.askopenfilename(title='Open stream-file')
    data = None

    # Opening file and parsing text
    with open(file, encoding='utf-8', errors='ignore') as f:
        text = f.readlines()
        if text[0] != 's16\n':
            raise Exception('Not a valid stream file!')
            return 0
        delim = ' '  # delimiter between the columns
        header = 2  # number of lines to skip in the beginning
        columns = (0, 1, 2)  # numbers of columns to get
        # Defult columns:
        # 0 – Dwell time
        # 1 – x-position
        # 2 – y-position
        f = open(file, encoding='utf-8', errors='ignore')
        print('Reading stream-file...')
        data = np.genfromtxt(f, dtype=np.float64, comments='#', delimiter=delim, skip_header=header, usecols=columns,
                             invalid_raise=False)
        print('Done!')

    # Determining volume dimensions with an offset
    # offset -= 1
    x_max, x_min = data[:, 1].max(), data[:, 1].min()
    x_dim = (x_max - x_min) * offset
    x_delta = (x_max - x_min) * (offset - 1) / 2
    y_max, y_min = data[:, 2].max(), data[:, 2].min()
    y_dim = (y_max - y_min) * offset
    y_delta = (y_max - y_min) * (offset - 1) / 2
    z_dim = max(x_dim, y_dim) * 2

    if x_dim < 1000 or y_dim < 1000: # checking if both dimensions are at least 100 nm
        if x_dim < 1000:
            x_dim = ((y_dim/2)//10)*10
            x_delta = x_dim/2
        if y_dim < 1000:
            y_dim = ((x_dim/2)//10)*10
            y_delta =  y_dim/2
    # Getting local coordinates
    data[:, 0] /= 1E7  # converting [0.1 µs] to [s]
    data[:, 1] -= x_min - x_delta
    data[:, 1] /= 10 # converting [0.1 nm] to [nm]
    # data[:, 1] += (x_dim - data[:, 1].max())/ 2 # shifting path center to the center of the volume
    data[:, 2] -= y_min - y_delta
    data[:, 2] /= 10
    # data[:, 2] += (y_dim - data[:, 2].max())/ 2 # shifting path center to the center of the volume
    data = np.roll(data, -1, 1)

    # Summing dwell time of consecutive points with same coordinates
    if collapse:
        p_d = np.diff(data[:, (0, 1)], axis=0).astype(bool)
        p_d = p_d[:, 0] + p_d[:, 1]
        collapsed_arr = []
        i = 0
        while i < p_d.shape[0] - 1:
            b = np.copy(data[i])
            while not p_d[i] and i < p_d.shape[0] - 1:
                b[2] += data[i, 2]
                i += 1
            collapsed_arr.append(b)
            i += 1
        data = np.asarray(collapsed_arr)

    return data, np.array([z_dim/10, y_dim/10, x_dim/10], dtype=int)


def generate_pattern(pattern, loops, dwell_time, x, y, params, step=1):
    """
    Generate a stream-file for a simple figure.

    :param pattern: name of a shape: point, line, square, rectangle, circle
    :param loops: amount of passes
    :param dwell_time: time spent on each point, s
    :param x: center x position of the figure, nm
    :param y: center y position of the figure, nm
    :param params: figure parameters, nm;
        (length) for line, (diameter) for circle, (edge length) for cube
    :param step: distance between each point, nm
    :return: array(x positions[nm], y positions[nm], dwell time[s])
    """
    pattern = pattern.casefold()
    path = None
    if pattern == 'point':
        path = generate_point(loops, dwell_time, x, y)
    if pattern == 'line':
        path = generate_line(loops, dwell_time, x, y, *params, step=step)
    if pattern == 'circle':
        path = generate_circle(loops, dwell_time, x, y, *params, step=step)
    if pattern == 'square':
        path = generate_square(loops, dwell_time, x, y, *params, step=step)
    if pattern == 'rectangle':
        path = generate_square(loops, dwell_time, x, y, *params, step=step)

    return path


def generate_point(loops, dwell_time, x, y):
    # loop = np.asarray([x, y, dwell_time]).reshape(1, 3)
    # path = np.tile(loop, (loops, 1))
    path = np.asarray([x, y, loops * dwell_time]).reshape(1, 3)
    return path


def generate_circle(loops, dwell_time, x, y, diameter, _, step=1):
    angle_step = step / diameter / 2
    n = int(np.pi * 2 // angle_step)
    loop = np.zeros((n, 3))
    stub = np.arange(angle_step, np.pi * 2, angle_step)
    loop[:, 0] = diameter / 2 * np.sin(stub) + y
    loop[:, 1] = diameter / 2 * np.cos(stub) + x
    loop[:, 2] = dwell_time
    path = np.tile(loop, (loops, 1))
    a = 0
    return path


def generate_square(loops, dwell_time, x, y, side_a, side_b=None, step=1):
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
    path = np.concatenate([edge_top, edge_right, edge_bottom, edge_left])
    path[:, 2] = dwell_time
    path = np.tile(path, (loops, 1))
    return path


def generate_line(loops, dwell_time, x, y, line, _, step=1):
    start = x - line / 2
    end = x + line / 2
    loop1 = np.arange(start, end+1e-8, step)
    loop2 = np.arange(loop1[-1], start-1e-8, -step)
    loop = np.concatenate([loop1, loop2])
    path = np.empty((loop1.shape[0]+loop2.shape[0], 3))
    path[:, 0] = loop
    path[:, 1] = y
    path[:, 2] = dwell_time
    path = np.tile(path, (loops, 1))
    return path
