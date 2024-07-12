"""
Module for continuous process data recording
"""
import random as rnd
import sys
import time
import timeit
from math import floor, log
from threading import Thread, Condition
from dataclasses import dataclass

import numpy as np
import pandas as pd

from febid.libraries.vtk_rendering.VTK_Rendering import save_deposited_structure


@dataclass
class SynchronizationHelper:
    """
    Secures a flag that serves as a signal to the threads that have it weather to stop execution or not.
    True to stop execution, False to continue.
    Also contains timer that counts intrinsic simulation time.
    """
    run_flag: bool
    loop_tick: Condition = Condition() # this allows the thread to pause instead of constantly looping
    _current_time: float = 0

    @property
    def timer(self):
        return self._current_time

    @timer.setter
    def timer(self, value):
        self._current_time = value

    def __repr__(self):
        return str(self.run_flag)

    def __bool__(self):
        return self.run_flag


class MonitoringDaemon(Thread):
    def __init__(self, run_flag: SynchronizationHelper, refresh_rate, purpose='Unidentified'):
        super().__init__()
        self.run_flag = run_flag
        self.refresh_rate = refresh_rate
        self.purpose = purpose
        self.start_time = timeit.default_timer()
        self.start_time_sim = run_flag.timer
        self.passed_time = run_flag.timer

    def run(self):
        print(f'Starting {self.purpose} daemon.')
        next_record_time = self.passed_time + self.refresh_rate
        self.run_flag.loop_tick.acquire()
        while not self.run_flag:
            self.run_flag.loop_tick.wait()
            if next_record_time < self.run_flag.timer:
                self.passed_time = self.run_flag.timer
                next_record_time = self.passed_time + self.refresh_rate
                self.looped_func()
        self.looped_func(end=True)
        self.run_flag.loop_tick.release()
        print(f'Closing {self.purpose} daemon.')

    def looped_func(self, end=False):
        """
        The function that is looped in the run() method. Should be overriden.
        """
        pass


class StructureSaver(MonitoringDaemon):
    """
    Class for saving structure data to vtk file.
    """
    def __init__(self, observed_obj, run_flag: SynchronizationHelper, refresh_rate, filename):
        super().__init__(run_flag, refresh_rate, purpose='Structure saver')
        self.observed_obj = observed_obj
        self.filename = filename

    def looped_func(self, end=False):
        pr = self.observed_obj
        structure = pr.structure
        sim_t = pr.t
        t = timeit.default_timer() - self.start_time
        beam_position = (pr.x0, pr.y0)
        save_deposited_structure(structure, sim_t, t, beam_position, self.filename)

# TODO: There is an Excel context manager in Pandas
class Statistics(MonitoringDaemon):
    """
    Class implementing statistics gathering and saving(to excel).

        Report contains following columns:

    Time, Time passed, Simulation time, Simulation speed, N of cells(filled), Volume, Min.precursor coverage, Growth rate

        It is possible to automatically include graphs into Excel files
        Additionally, initial simulation parameters are added to 3 separate sheets
    """

    def __init__(self, observed_obj, run_flag: SynchronizationHelper, refresh_rate, filename=f'run_id{rnd.randint(100000, 999999)}',
                 record_interval=1e-3):
        super().__init__(run_flag, refresh_rate, purpose='Statistics gathering')
        self.observed_obj = observed_obj
        self.filename = filename + '.xlsx'
        self.sheet_name = 'Data'
        self.columns = ['Time', 'Time passed', 'Sim.time', 'Min.precursor coverage', 'Volume', 'Max. temperature', ]
        self.units = ['', 's', 's', '', '', '1/s', '1/s']
        self.data = pd.DataFrame(columns=self.columns)
        self.data.loc[0] = [pd.Timestamp.now(), 0, 0, 0, 0, 0]
        self.step = self.data.copy()
        self.parameters = []
        self.parameters_units = []
        self.writer = None
        self.save_freq = 10  # seconds
        self.record_interval = record_interval
        self.last_row = 0  # last row recorded previously
        self.time = timeit.default_timer()

        # Creating new file, old file is overwritten
        filename = self.filename
        self.data.to_excel(filename, startrow=self.last_row, sheet_name=self.sheet_name, header=True)
        self.last_row = 1

    def add_stat(self, name, first_value=0):
        """
        Add a new statistic to the table.
        It is recorded in monitoring function and how it is collected is up to the user.
        """
        self.data[name] = first_value
        self.columns.append(name)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def shape(self):
        return self.data.shape

    def get_params(self, arg: dict, name: str):
        """
        Collect initial parameters and save them to Excel-file

        :param arg: a dictionary of parameters
        :param name: a name for the provided parameters
        :return:
        """
        series = pd.Series(arg)
        series.name = name
        self.parameters.append(series)
        try:
            args, kwargs = self.__get_writer_args_and_kwargs()
            with pd.ExcelWriter(*args, **kwargs) as writer:
                series.to_excel(writer, sheet_name=name)
            # self.writer.save()
        except Exception as e:
            print(f'Failed to save setup parameters to excel file: {e.args}')

    def append(self, *stats):
        """
        Add a new record to the statistics.
        The number of stats must include manually added ones

        :param stats: current simulation time, current number of deposited cells and manually added columns
        :return:
        """
        self.dt = 0
        self.av_temperature = 0
        record = {}
        cols = self.columns
        try:
            time_now = pd.Timestamp.now()
            record[cols[0]] = time_now
            record[cols[1]] = (time_now - self.data.at[0, cols[0]]).total_seconds()
            for i in range(len(stats)):
                record[cols[i + 2]] = stats[i]
            self.data.loc[self.shape[0]] = tuple([record[cols[i]] for i in range(len(cols))])
        except Exception as e:
            print('An error occurred while recording statistics.')
            print(e.args)

    def plot(self, x, y):
        """
        ['Time', 'Sim.time', 'Sim.speed', 'Volume', Min.precursor coverage', 'Growth rate']
        :param x:
        :param y:
        :return:
        """
        if x not in self.columns or y not in self.columns:
            print('Column with this name does not exist!')
            return
        self.plot(x=x, y=y)

    def save_to_file(self, force=False):
        """
        Write collected statistics to an Excel file.
        The gathered statistics are appended to the end of the table every couple of seconds
        Caution: the session keeps the file open until it finishes.
        """

        def write_to_file(data, header, last_row):
            args, kwargs = self.__get_writer_args_and_kwargs()
            with pd.ExcelWriter(*args, **kwargs) as writer:
                data.to_excel(writer, startrow=last_row, sheet_name=self.sheet_name, header=header)
                self.last_row = writer.sheets[self.sheet_name].max_row

        if timeit.default_timer() - self.time <= self.save_freq and not force:
            return
        else:
            self.time = timeit.default_timer()
        if force:
            last_row = 0
            header = True
        else:
            last_row = self.last_row
            header = False
        data = self.data.iloc[last_row:]
        while True:
            try:
                write_to_file(data, header, last_row)
                break
            except PermissionError as e:
                print(f'Was unable to save statistics to file, the following error occurred: {e.args}')
                input('Please close the file and press Enter to continue recording.')

    def get_growth_rate(self):
        delta = 4
        t = self.data['Sim.time']
        vol = self.data['Volume']
        gr = np.zeros_like(t)
        for i in range(delta, t.shape[0]):
            gr[i] = (vol[i] - vol[i - delta]) / (t[i] - t[i - delta])
        gr[gr == 0] = np.nan
        self.data['Growth rate'] = gr

    def add_plots(self, *args, position='J1'):
        """
        Add scatter plots to the Excel-file.

            Args is a list of tuples of column names to be plotted: [(x1, y1), (x2, y2)]
            Position is a list of cells where to put the graphs (by the upper-left corner)
        """
        writer = pd.ExcelWriter(self.filename, engine='xlsxwriter', )
        self.data.to_excel(writer, sheet_name=self.sheet_name)

        for arg, pos in zip(*args, position):
            self.add_plot(*arg, writer, pos)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        writer.close()

    def add_plot(self, x, y, writer, position='J1'):
        def mag(val):  # define magnitude of the number
            return abs(floor(log(abs(val), 10)))

        # workbook:xlsxwriter.Workbook = writer.book # debug:typing reveals chart methods
        workbook = writer.book
        worksheet = writer.sheets[self.sheet_name]

        df = self.data

        # Create a chart object.
        chart = workbook.add_chart({'type': 'scatter'})

        # Configure the series of the chart from the dataframe data.
        max_row = len(df)
        x_i = df.columns.to_list().index(x) + 1
        y_i = df.columns.to_list().index(y) + 1
        chart.add_series({
            'name': [self.sheet_name, 0, y_i],
            'categories': [self.sheet_name, 1, x_i, max_row, x_i],
            'values': [self.sheet_name, 1, y_i, max_row, y_i],
            'marker': {'type': 'circle', 'size': 1,
                       'border': {'color': '#004586'},
                       'fill': {'color': '#004586'}, },
            'line': {'none': True},
        })
        chart.set_legend({'none': True})
        chart.chart_name = y

        # Define axis scale to include a 10% margin
        x_min = df[x].min()
        ax_min = np.round(x_min * 1.1, mag(x_min) + 1) if x_min != 0 else x_min
        x_max = df[x].max()
        ax_max = np.round(x_max * 1.1, mag(x_max) + 1) if x_max != 0 else x_max
        y_min = df[y].min()
        ay_min = np.round(y_min * 1.1, mag(y_min) + 1) if y_min != 0 else y_min
        y_max = df[y].max()
        ay_max = np.round(y_max * 1.1, mag(y_max) + 1) if y_max != 0 else y_max

        # Define major ticks
        ax_major = np.round((x_max - x_min) / 10, mag((x_max - x_min) / 10) - 1)
        ay_major = np.round((y_max - y_min) / 10, mag((y_max - y_min) / 10) - 1)
        # Configure the chart axes.
        chart.set_x_axis({'name': x,
                          'min': ax_min,
                          'max': ax_max,
                          'major_unit': ax_major,
                          'major_gridlines': {'visible': True, 'line': {'color': '#B3B3B3'}}, })
        chart.set_y_axis({'name': y,
                          'min': ay_min,
                          'max': ay_max,
                          'major_unit': ay_major,
                          'major_gridlines': {'visible': True, 'line': {'color': '#B3B3B3'}}, })
        chart.set_size({'x_scale': 1.2, 'y_scale': 1.5})
        chart.set_title({'name': y})

        # Insert the chart into the worksheet.
        worksheet.insert_chart(position, chart)

    def looped_func(self, end=False):
        pr = self.observed_obj
        self.append(pr.t, pr.min_precursor_coverage, pr.dep_vol, pr.max_T, )
        self.save_to_file(end)

    def __get_writer_args_and_kwargs(self):
        return (self.filename,), dict(engine='openpyxl', mode='a', if_sheet_exists='overlay')
