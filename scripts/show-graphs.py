#!/usr/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas
import sys


def main(filename, log=False):

    print("Loading file '{0}'...".format(filename), end='', flush=True)
    df = pandas.read_csv(filename)
    print('finished.', flush=True)
    data = list(zip(df['S304'], df['E304'], df['I304'], df['R304'], df['P304']))
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
