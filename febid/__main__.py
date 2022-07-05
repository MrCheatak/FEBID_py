import sys

import febid
from febid.start import start_default

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ['show_file', 'show_animation']:
            if sys.argv[1] == 'show_file':
                febid.show_file.show_structure()
            if sys.argv[1] == 'show_animation':
                febid.show_animation.show_animation()
        else:
            print(f'Unexpected argument {sys.argv[1]}')
    else:
        start_default()
