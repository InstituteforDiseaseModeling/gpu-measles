#!/usr/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas
import sys


'''
"interesting" locales:
         name                 node id* (node index)
    NW - KANO:NASARAWA             314 (313)
    SW - LAGOS:ALIMOSHO            656 (655)
    SS - RIVERS:PORT_HARCOURT      636 (635)
    NC - FCT_ABUJA:ABUJA_MUNICIPAL  25 ( 24)
    NE - BORNO:MAIDUGUI            183 (182)
    SE - ANAMBRA:IDEMILI_NORTH     446 (445)
    *from demographics
'''

def main(filename, log=False):

    print("Loading file '{0}'...".format(filename), end='', flush=True)
    df = pandas.read_csv(filename)
    print('finished.', flush=True)
    data = list(zip(df['S314'], df['E314'], df['I314'], df['R314'], df['P314']))
    if log:
        plt.yscale('log')
    plt.plot(data)
    plt.show()

    return


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-l', '--log', default=False, action='store_true')

    args = parser.parse_args()

    main(args.filename, args.log)
    sys.exit(0)
