The interface is built using PyQt5.
Interface is divided into several files corresponding to windows implemented.
The source files for every window are the .ui files. They are created in 'QT Designer'
software and it is recommended to use it for further interface development.

The interface files, though, have to be 'compiled' into Python code.
Every time an interface file is changed, it has to be compiled to reflect the changes in the program.
In order to do this, the following command needs to be executed in the terminal:

pyuic5 -o main_window.py main_window.ui

It takes the 'main_window.ui' interface file and compiles it in-place into 'main_window.py' Python file.
Keep in mind that you may have to navigate to the folder containing the file first.
After that, you are good to go to launch the program.