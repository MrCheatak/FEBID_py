# This is a sample Python script.
# import Process
import Tests
import numpy as np
from timebudget import timebudget
import itertools
import cProfile

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    """
    Prints stuff

    :param name: string to print
    :return: Output to terminal
    """
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #cProfile.runctx('Process', globals(),locals())
    #cProfile.runctx('main',globals(),locals())
    # exec("Process")
    exec("Tests")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
