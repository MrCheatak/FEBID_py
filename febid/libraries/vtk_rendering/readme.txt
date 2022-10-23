Show_file.py script can be configured as a default application for .vtk-files in Linux after installing febid package.
The general idea is to create a shell script that activates python environment and runs a launching command in it.

1. Create a text file and substitute dummies with your paths:

#!/bin/bash
# The second argument is the path of the specified file or the file opened by double clicking
path=$1
echo $path
# Activating python environment
source "/PathToPythonEnvironment/bin/activate"

python --version
# Running show_file with the file opened
python -m febid show_file $path

$SHELL

2. Change extension to .sh and make file executable, in terminal:
    chmod u+x YourScriptFileName.sh
4. The resulting file can be assigned as a default application in 'Open with..' context menu. The file can then be
opened by double-clicking. At least in Linux Cinnamon, several files can be viewed at the same time.


The same behaviour can be configured on Macs with the standard Automator app.

1. Create an application and add "Run shell script" action. It is important that the executable is created as
    an app and not as a script, so that it can receive files as an input.
2. Set Pass input to 'as arguments' and paste the following snippet substituting the path to your Python environment:

. /PathToPythonEnvironment/bin/activate

python --version
python -m febid show_file $1 # $1 is the path to the file that is being opened

3. Save the application. The resulting app can be used to open .vtk-files from the simulation and can be assigned
    as default app for .vtk-files. Only one file can be opened by the script at the same time.