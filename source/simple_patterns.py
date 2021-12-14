"""
Stream file generator
"""
import tkinter.filedialog as fd
import numpy as np


def open_stream_file(file='', offset=1.5):
    """
    Open stream file and define enclosing array bounds

    :param file: path to the stream-file
    :param offset: determines a margin around the printing path
    :return: normalized directives, dimensions of the enclosing volume
    """
    if not file:
        file = fd.askopenfilename()
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

    # Determining chamber dimensions
    offset -= 1
    x_max, x_min = data[:, 1].max(), data[:, 1].min()
    x_dim = x_max - x_min
    x_min -= x_dim * offset
    x_max += x_dim * offset
    x_dim = x_max - x_min
    y_max, y_min = data[:, 2].max(), data[:, 2].min()
    y_dim = y_max - y_min
    y_min -= y_dim * offset
    y_max += y_dim * offset
    y_dim = y_max - y_min
    z_dim = max(x_dim, y_dim) * 2

    # Getting local coordinates
    data[:, 0] /= 1E10  # converting [0.1 ns] to [s]
    data[:, 1] -= x_min
    data[:, 2] -= y_min
    data = np.roll(data, -1, 1)

    return data, (z_dim, y_dim, x_dim)


def generate_pattern(pattern, loops, dwell_time, x, y, params, step=1):
    """
    Generate stream file - like directions for printing one of the simple patterns

    :param pattern: name of a shape: point, line, square, rectangle, circle
    :param loops: amount of passes
    :param dwell_time: time spent on each point
    :param x: center x position of the shape
    :param y: center y position of the shape
    :param params: shape parameters
    :param step: distance between each point
    :return:
    """
    path = None
    if pattern == 'point':
        path = generate_point(loops, dwell_time, x, y)
    if pattern == 'line':
        path = generate_line(loops, dwell_time, x, y, *params)
    if pattern == 'circle':
        path = generate_circle(loops, dwell_time, x, y, *params)
    if pattern == 'square':
        path = generate_square(loops, dwell_time, x, y, *params)
    if pattern == 'rectangle':
        path = generate_square(loops, dwell_time, x, y, *params)

    return path


def generate_point(loops, dwell_time, x, y):
    loop = np.asarray([x, y, dwell_time]).reshape(1, 3)
    path = np.tile(loop, (loops, 1))
    return path


def generate_circle(loops, dwell_time, x, y, radius, step=1):
    angle_step = step / radius
    n = int(np.pi * 2 // angle_step)
    loop = np.zeros((n, 3))
    stub = np.arange(angle_step, np.pi * 2, angle_step)
    loop[:, 0] = radius * np.sin(stub) + y
    loop[:, 1] = radius * np.cos(stub) + x
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


def generate_line(loops, dwell_time, x, y, line, step=1):
    start = x - line / 2
    end = x + line / 2
    path = np.empty((int(line / step * 2) - 2, 3))
    loop1 = np.arange(start + step, end, step)
    loop2 = np.arange(end - step, start, -step)
    loop = np.concatenate([loop1, loop2])
    path[:, 0] = y
    path[:, 1] = loop
    path[:, 2] = dwell_time
    path = np.tile(path, (loops, 1))
    return path
