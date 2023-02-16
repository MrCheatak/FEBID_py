"""
Module for continuous process data recording
"""
import random as rnd
import sys
import timeit
from math import floor, log

import numpy as np
import pandas as pd


class Statistics():
    """
    Class implementing statistics gathering and saving(to excel).

        Report contains following columns:

    Time, Time passed, Simulation time, Simulation speed, N of cells(filled), Volume, Min.precursor coverage, Growth rate

        It is possible to automatically include graphs into Excel files
        Additionally, initial simulation parameters are added to 3 separate sheets
    """

    def __init__(self, filename=f'run_id{rnd.randint(100000, 999999)}'):
        self.filename = filename + '.xlsx'
        self.sheet_name = 'Data'
        self.columns = ['Time', 'Time passed', 'Sim.time', 'Min.precursor coverage', 'Volume', 'Max. temperature',]
        # self.units = ['', 's', 's', '', '', '1/s', '1/s']
        self.data = pd.DataFrame(columns=self.columns)
        self.data.loc[0] = [pd.Timestamp.now(), 0, 0, 0, 0, 0]
        self.step = self.data.copy()
        self.parameters = []
        self.parameters_units = []
        self.writer = None
        self.save_freq = 10 # seconds
        self.last_row = 0 # last row recorded previously
        self.time = timeit.default_timer()

        # Creating new file, old file is overwritten
        filename = self.filename
        self.data.to_excel(filename, startrow=self.last_row, sheet_name=self.sheet_name, header=True)
        self.last_row = 1
        self.writer = pd.ExcelWriter(self.filename, engine='openpyxl', mode='a', if_sheet_exists='overlay')

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
            series.to_excel(self.writer, sheet_name=name)
            self.writer.save()
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
                record[cols[i+2]] = stats[i]
            # time_now = pd.Timestamp.now()
            # sim_time = stats[2]
            # time_passed = (stats[0] - self.data.at[0, cols[0]]).total_seconds()
            # sim_speed = stats[1] / time_passed
            # growth_speed = stats[2] / time_passed * 60 * 60
            # growth_rate = stats[2] / stats[1]
            # self.step = pd.Series({cols[1]:stats[0], cols[3]:stats[1]}, name=pd.Timestamp.now())
            # self.step.loc[self.shape[0]] = (stats[0], time_passed, stats[1], sim_speed, stats[2], growth_speed, growth_rate, stats[3])
            self.data.loc[self.shape[0]] = tuple([record[cols[i]] for i in range(len(cols))])
        except Exception as e:
            print('An error occurred while recording statistics.')
            print(e.args)

        # self.data = self.data.append(self.step) # DataFrame.append() is not an in-place method like list.append()

    def plot(self, x, y):
        """
        ['Time', 'Sim.time', 'Sim.speed', 'Volume', Min.precursor coverage', 'Growth rate']
        :param x:
        :param y:
        :return:
        """
        if x not in self.columns or y not in self.columns:
            print(f'Column with this name does not exist!')
            return
        self.plot(x=x, y=y)

    def save_to_file(self, force=False):
        """
        Write collected statistics to an Excel file.
        The gathered statistics are appended to the end of the table every couple of seconds
        Caution: the session keeps the file open until it finishes.
        """
        if not timeit.default_timer()-self.time > self.save_freq and not force:
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
        try:
            data.to_excel(self.writer, startrow=last_row, sheet_name=self.sheet_name, header=header)
            self.writer.save()
            self.last_row = self.writer.sheets[self.sheet_name].max_row
        except Exception as e:
            print(f'Was unable to save statistics to file, the following error occurred: {e.args}')
            sys.exit()

    def __get_time_passed(self):
        self.data.iloc[1:, self.columns[1]] = self.data.loc[1:, 'Time'] - self.data.loc[:-1, 'Time']

    def __get_sim_speed(self):
        self.data.iloc['Sim.speed'] = self.data['Sim.time'] / self.data['Time']

    def get_growth_rate(self):
        delta = 4
        t = self.data['Sim.time']
        vol = self.data['Volume']
        gr = np.zeros_like(t)
        for i in range(delta, t.shape[0]):
            gr[i] = (vol[i] - vol[i-delta]) / (t[i] - t[i-delta])
        gr[gr==0] = np.nan
        self.data['Growth rate'] = gr


    def add_plots(self, *args, position='J1'):
        """
        Add scatter plots to the Excel-file.

            Args is a list of tuples of column names to be plotted: [(x1, y1), (x2, y2)]
            Position is a list of cells where to put the graphs (by the upper-left corner)
        """
        if self.writer:
            self.writer.save()
            self.writer.close()
            self.writer = None
        writer = pd.ExcelWriter(self.filename, engine='xlsxwriter',)
        self.data.to_excel(writer, sheet_name=self.sheet_name)

        for arg, pos in zip(*args, position):
            self.add_plot(*arg, writer, pos)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        writer.close()
        # if not self.writer:
        #     self.writer = pd.ExcelWriter(self.filename, engine='openpyxl', mode='a', if_sheet_exists='overlay')

    def add_plot(self, x, y, writer, position='J1'):
        def mag(val): # define magnitude of the number
            return abs(floor(log(abs(val), 10)))
        # workbook:xlsxwriter.Workbook = writer.book # debug:typing reveals chart methods
        workbook = writer.book
        worksheet = writer.sheets[self.sheet_name]

        df = self.data

        # Create a chart object.
        chart = workbook.add_chart({'type': 'scatter'})

        # Configure the series of the chart from the dataframe data.
        max_row = len(df)
        x_i = df.columns.to_list().index(x)+1
        y_i = df.columns.to_list().index(y)+1
        chart.add_series({
            'name': [self.sheet_name, 0, y_i],
            'categories': [self.sheet_name, 1, x_i, max_row, x_i],
            'values': [self.sheet_name, 1, y_i, max_row, y_i],
            'marker': { 'type': 'circle', 'size': 1,
                        'border': {'color': '#004586'},
                        'fill': {'color': '#004586'}, },
            'line': {'none': True},
        })
        chart.set_legend({'none': True})
        chart.chart_name = y

        # Define axis scale to include a 10% margin
        x_min = df[x].min()
        ax_min = np.round(x_min * 1.1, mag(x_min)+1) if x_min != 0 else x_min
        x_max = df[x].max()
        ax_max = np.round(x_max * 1.1, mag(x_max)+1) if x_max != 0 else x_max
        y_min = df[y].min()
        ay_min = np.round(y_min * 1.1, mag(y_min)+1) if y_min != 0 else y_min
        y_max = df[y].max()
        ay_max = np.round(y_max * 1.1, mag(y_max)+1) if y_max != 0 else y_max

        # Define major ticks
        ax_major = np.round((x_max - x_min) / 10, mag((x_max - x_min) / 10) - 1)
        ay_major = np.round((y_max - y_min) / 10, mag((y_max - y_min) / 10) - 1)
        # Configure the chart axes.
        chart.set_x_axis({'name': x,
                          'min': ax_min,
                          'max': ax_max,
                          'major_unit': ax_major,
                          'major_gridlines': {'visible': True, 'line': {'color':'#B3B3B3'}},})
        chart.set_y_axis({'name': y,
                          'min': ay_min,
                          'max': ay_max,
                          'major_unit': ay_major,
                          'major_gridlines': {'visible': True, 'line': {'color':'#B3B3B3'}},})
        chart.set_size({'x_scale': 1.2, 'y_scale': 1.5})
        chart.set_title({'name': y})

        # Insert the chart into the worksheet.
        worksheet.insert_chart(position, chart)

    def __del__(self):
        try:
            self.writer.save()
            self.writer.close()
            self.writer = None
        except UserWarning:
            pass
        except AttributeError:
            pass