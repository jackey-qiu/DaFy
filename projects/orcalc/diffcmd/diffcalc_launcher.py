#!/usr/bin/python

import argparse
import subprocess
import os
import getpass

DIFFCALC_BIN = os.path.split(os.path.realpath(__file__))[0]
DIFFCALC_ROOT = os.path.abspath(os.path.join(DIFFCALC_BIN, os.pardir))

MODULE_FOR_MANUALS = '_make_sixcircle_manual'

def main():
    parser = argparse.ArgumentParser(description='Diffcalc: A diffraction condition calculator of x-ray and neutron crystalography')
    parser.add_argument('--modules', dest='show_modules', action='store_true',
                        help='list available modules')
    parser.add_argument('--python', dest='use_python', action='store_true',
                        help='run within python rather than ipython')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='run in debug mode')
    parser.add_argument('--make-manuals-source', dest='make_manuals', action='store_true',
                        help='make .rst manual files by running template through sixcircle')
    parser.add_argument('--non-interactive', dest='non_interactive', action='store_true',
                        help='do not enter interactive mode after startup')
    parser.add_argument('module', type=str, nargs='?',
                        help='the module to startup with')
    args = parser.parse_args()

    # Create list of available modules
    module_names = []
    for module_path in os.listdir(os.path.join(DIFFCALC_ROOT, 'startup')):
        if not module_path.startswith('_') and module_path.endswith('.py'):
            module_names.append(module_path.split('.')[0])
    module_names.sort()

    if args.show_modules:
        print_available_modules(module_names)
        exit(0)

    if not args.make_manuals and not args.module:
        print("A module name should be provided. Choose one of:")
        print_available_modules(module_names)
        exit(0)

    if args.make_manuals:
        if args.module:
            print("When building the manuals no module should be given")
            exit(1)
        args.module = MODULE_FOR_MANUALS

    if not args.make_manuals and args.module not in module_names:
        print("The provided argument '%s' is not one of:" % args.module)
        print_available_modules(module_names)
        exit(1)

    env = os.environ.copy()

    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = ''
    env['PYTHONPATH'] = DIFFCALC_ROOT + ':' + env['PYTHONPATH']

    diffcmd_start_path = os.path.join(DIFFCALC_ROOT, 'diffcmd', 'start.py')

    if args.use_python:
        cmd = 'python3'
    else:  # ipython
        cmd = 'ipython3 --no-banner --HistoryManager.hist_file=/tmp/ipython_hist_%s.sqlite' % getpass.getuser()

    iflag = '' if args.non_interactive else '-i'
    cmd = cmd + ' ' + ' '.join([iflag, diffcmd_start_path, args.module, str(args.debug)])

    print('Running: ' + cmd)
    rc = subprocess.call(cmd, env=env, shell=True)
    exit(rc)


def print_available_modules(module_names):
    lines = []
    for m in sorted(module_names):
        lines.append('   ' + m)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
#
