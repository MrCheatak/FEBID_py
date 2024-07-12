"""
Package main entry point
"""
import sys

import febid
from febid.ui import ui_shell
from febid.start import Starter

intro = ('Welcome to the FEBID simulator. \n'
         ' To start a basic session with GUI use \n'
         ' \t `python -m febid` \n'
         ' To start a GUI session with predefined configuration use \n'
         ' \t `python -m febid ui <configuration_file>` \t(an example can be found in the repo) \n'
         ' GUI sessions still use console for output. \n'
         ' To start a session without a UI use \n'
         ' \t `python -m febid no_ui <configuration_file>` \t this command will start the simulation right away. \n'
         ' To open a single 3D .vtk-file use \n'
         ' \t `python -m febid show_file <full_filepath>` \t if a file is not specified, a file prompt window will open. \n'
         ' To view an animated series from .vtk-files use \n'
         ' \t `python -m febid show_animation <directory>` \t if a directory is not specified, a directory prompt window will open. \n')


def start_ui(config_f=None):
    """
    Start FEBID with graphical user interface.

    :param config_f: configuration file
    :return:
    """
    ui_shell.start(config_f)


def start_no_ui(config_f=None):
    """
    Start FEBID without graphical user interface.

    :param config_f: configuration file
    :return:
    """
    Starter(config_f).start()


def welcome():
    """
    This function prints the intro and performs actions based on command line arguments.
    """
    print(intro)
    command_functions = {
        'show_file': febid.show_file.show_structure,
        'show_animation': febid.show_animation.render_animation,
        'gui': start_ui,
        'no_gui': Starter().start
    }
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command in command_functions:
            function = command_functions[command]
            if len(sys.argv) > 2:
                function(sys.argv[2])
            else:
                function()
        else:
            print(f'Unexpected argument {command}')
    else:
        start_ui()


if __name__ == '__main__':
    welcome()
