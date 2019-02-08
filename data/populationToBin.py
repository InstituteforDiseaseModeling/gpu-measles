#!/usr/bin/python3

from argparse import ArgumentParser
import numpy
import sys
import NGA_population as ngapop


def main(filename):

    print('Writing population values to {0}'.format(filename))
    keys = sorted([key for key in ngapop.dataDic.keys() if key.count(':') == 4])
    populations = [ngapop.dataDic[key][0] for key in keys]
    populations = numpy.array(populations, dtype=numpy.uint32)
    populations.tofile(filename)

    return


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-o', '--outfile', default='nga-populations-2015.bin')

    args = parser.parse_args()

    main(args.outfile)

    sys.exit(0)
