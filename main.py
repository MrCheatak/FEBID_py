# This is a sample Python script.
#import Process
import numpy as np

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    substrate = np.zeros((10, 10, 10, 2), dtype=np.float)
    substrate[:, :, :, 1] = 2
    #exec("Process")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
