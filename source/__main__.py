import sys

import source
from source.start import start_default

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ['show_file', 'show_animation']:
            if sys.argv[1] == 'show_file':
                source.show_file.show_structure()
            if sys.argv[1] == 'show_animation':
                source.show_animation.show_animation()
        else:
            print(f'Unexpected argument {sys.argv[1]}')
    else:
        start_default()
