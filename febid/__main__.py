"""
Package main entry point
"""
import sys

import febid
from febid.start import start_default, start_no_ui

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

if __name__ == '__main__':
    print(intro)
    if len(sys.argv) > 1:
        if sys.argv[1] in ['show_file', 'show_animation', 'gui', 'no_gui']:
            if sys.argv[1] == 'show_file':
                file = None
                try:
                    file = sys.argv[2]
                except:
                    pass
                febid.show_file.show_structure(file)
            if sys.argv[1] == 'show_animation':
                directory = None
                try:
                    directory = sys.argv[2]
                except:
                    pass
                febid.show_animation.show_animation(directory)
            if sys.argv[1] == 'gui':
                if len(sys.argv) > 2:
                    start_default(sys.argv[2])
                else:
                    start_default()
            if sys.argv[1] == 'no_gui':
                if len(sys.argv) < 2:
                    print(f'Please specify configuration file when starting without GUI')
                else:
                    start_no_ui(sys.argv[2])
        else:
            print(f'Unexpected argument {sys.argv[1]}')
    else:
        start_default()
